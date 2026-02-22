import spacy
import bert_score
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from selfcheckgpt.utils import MQAGConfig, expand_list1, expand_list2, NLIConfig, LLMPromptConfig
from selfcheckgpt.modeling_mqag import question_generation_sentence_level, answering
from selfcheckgpt.modeling_ngram import UnigramModel, NgramModel

# ---------------------------------------------------------------------------------------- #
# Functions for counting
def method_simple_counting(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def answerability_scoring(
    u_model,
    u_tokenizer,
    question,
    context,
    max_length,
    device,
):
    """
    :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
    """
    input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
    inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob

class SelfCheckMQAG:
    """
    SelfCheckGPT (MQAG varaint): Checking LLM's text against its own sampled texts via MultipleChoice Question Answering
    """
    def __init__(
        self,
        g1_model: str = None,
        g2_model: str = None,
        answering_model: str = None,
        answerability_model: str = None,
        device = None
    ):

        g1_model = g1_model if g1_model is not None else MQAGConfig.generation1_squad
        g2_model = g2_model if g2_model is not None else MQAGConfig.generation2
        answering_model = answering_model if answering_model is not None else MQAGConfig.answering
        answerability_model = answerability_model if answerability_model is not None else MQAGConfig.answerability

        # Question Generation Systems (G1 & G2)
        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)

        # Question Answering System (A)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(answering_model)
        self.a_model = LongformerForMultipleChoice.from_pretrained(answering_model)

        # (Un)Answerability System (U)
        self.u_tokenizer = LongformerTokenizer.from_pretrained(answerability_model)
        self.u_model = LongformerForSequenceClassification.from_pretrained(answerability_model)

        self.g1_model.eval()
        self.g2_model.eval()
        self.a_model.eval()
        self.u_model.eval()

        if device is None:
            device = torch.device("cpu")
        self.g1_model.to(device)
        self.g2_model.to(device)
        self.a_model.to(device)
        self.u_model.to(device)
        self.device = device
        print("SelfCheck-MQAG initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
        num_questions_per_sent: int = 5,
        scoring_method: str = "bayes_with_alpha",
        **kwargs,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param passage: str -- the passage to be evaluated, note that splitting(passage) ---> sentences
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param num_questions_per_sent: int -- number of quetions to be generated per sentence
        :return sent_scores: sentence-level score of the same length as len(sentences) # inconsistency_score, i.e. higher means likely hallucination
        """
        assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
        num_samples = len(sampled_passages)
        sent_scores = []
        for sentence in sentences:

            # Question + Choices Generation
            questions = question_generation_sentence_level(
                self.g1_model, self.g1_tokenizer,
                self.g2_model, self.g2_tokenizer,
                sentence, passage, num_questions_per_sent, self.device)

            # Answering
            scores = []
            max_seq_length = 4096 # answering & answerability max length
            for question_item in questions:
                question, options = question_item['question'], question_item['options']
                # response
                prob = answering(
                    self.a_model, self.a_tokenizer,
                    question, options, passage,
                    max_seq_length, self.device)

                u_score = answerability_scoring(
                    self.u_model, self.u_tokenizer,
                    question, passage,
                    max_seq_length, self.device)

                prob_s = np.zeros((num_samples, 4))
                u_score_s = np.zeros((num_samples,))
                for si, sampled_passage in enumerate(sampled_passages):

                    # sample
                    prob_s[si] = answering(
                        self.a_model, self.a_tokenizer,
                        question, options, sampled_passage,
                        max_seq_length, self.device)
                    u_score_s[si] = answerability_scoring(
                        self.u_model, self.u_tokenizer,
                        question, sampled_passage,
                        max_seq_length, self.device)

                # doing comparision
                if scoring_method == 'counting':
                    score = method_simple_counting(prob, u_score, prob_s, u_score_s, num_samples, AT=kwargs['AT'])
                elif scoring_method == 'bayes':
                    score = method_vanilla_bayes(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'], AT=kwargs['AT'])
                elif scoring_method == 'bayes_with_alpha':
                    score = method_bayes_with_alpha(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'])
                scores.append(score)
            sent_score = np.mean(scores)
            sent_scores.append(sent_score)

        return np.array(sent_scores)

class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self, default_model="en", rescale_with_baseline=True):
        """
        :default_model: model for BERTScore
        :rescale_with_baseline:
            - whether or not to rescale the score. If False, the values of BERTScore will be very high
            - this issue was observed and later added to the BERTScore package,
            - see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model # en => roberta-large
        self.rescale_with_baseline = rescale_with_baseline
        print("SelfCheck-BERTScore initialized")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            P, R, F1 = bert_score.score(
                    cands, refs,
                    lang=self.default_model, verbose=False,
                    rescale_with_baseline=self.rescale_with_baseline,
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent

class SelfCheckNgram:
    """
    SelfCheckGPT (Ngram variant): Checking LLM's text against its own sampled texts via ngram model
    Note that this variant of SelfCheck score is not bounded in [0.0, 1.0]
    """
    def __init__(self, n: int, lowercase: bool = True):
        """
        :param n: n-gram model, n=1 is Unigram, n=2 is Bigram, etc.
        :param lowercase: whether or not to lowercase when counting n-grams
        """
        self.n = n
        self.lowercase = lowercase
        print(f"SelfCheck-{n}gram initialized")

    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
    ):
        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be integer >= 1")
        ngram_model.add(passage)
        for sampled_passge in sampled_passages:
            ngram_model.add(sampled_passge)
        ngram_model.train(k=0)
        ngram_pred = ngram_model.evaluate(sentences)
        return ngram_pred

class SelfCheckNLI:
    """
    SelfCheckGPT (NLI variant): Checking LLM's text against its own sampled texts via DeBERTa-v3 finetuned to Multi-NLI
    Extended version with predict_matrix method that returns entailment, neutral, and contradiction probabilities.
    """
    def __init__(
        self,
        nli_model: str = None,
        device = None,
        batch_size: int = 32
    ):
        nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Map label indices to label names using model config
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Find indices for entailment, neutral, contradiction (case-insensitive)
        self.entailment_idx = None
        self.neutral_idx = None
        self.contradiction_idx = None
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if 'entail' in label_lower:
                self.entailment_idx = idx
            elif 'neutral' in label_lower:
                self.neutral_idx = idx
            elif 'contradict' in label_lower:
                self.contradiction_idx = idx
        
        # Validate we found the expected labels
        if self.entailment_idx is None or self.contradiction_idx is None:
            raise ValueError(f"Could not find entailment/contradiction labels in model. Available labels: {list(self.id2label.values())}")
        
        print(f"SelfCheck-NLI initialized to device {device}")
        neutral_info = f"N={self.id2label[self.neutral_idx]}({self.neutral_idx})" if self.neutral_idx is not None else "N=None"
        print(f"Label mapping: E={self.id2label[self.entailment_idx]}({self.entailment_idx}), "
              f"{neutral_info}, "
              f"C={self.id2label[self.contradiction_idx]}({self.contradiction_idx})")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is P(contradiction|sentence, sample)
        Uses E/C renormalization (drops neutral), matching the original paper implementation.
        """
        _, _, contra_scores = self.predict_matrix(sentences, sampled_passages)
        scores_per_sentence = contra_scores.mean(axis=-1)
        return scores_per_sentence

    @torch.no_grad()
    def predict_matrix(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        Return per-(sentence, sample) entailment and contradiction probabilities.
        Uses batched processing for efficiency. Always uses E/C renormalization (drops neutral).
        :param sentences: list[str] -- sentences to be evaluated
        :param sampled_passages: list[str] -- stochastically generated responses
        Returns: (entailment_scores, neutral_scores, contradiction_scores)
        Each has shape: (num_sentences, num_samples). neutral_scores is always None (neutral is ignored).
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        
        # Handle empty inputs
        if num_sentences == 0 or num_samples == 0:
            entailment_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
            neutral_scores = None  # Always None, neutral is ignored
            contradiction_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
            return entailment_scores, neutral_scores, contradiction_scores
        
        # Build all (sentence, sample) pairs
        pairs = []
        pair_indices = []  # Track which (sent_i, sample_i) each pair corresponds to
        
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                pairs.append((sentence, sample))
                pair_indices.append((sent_i, sample_i))
        
        # Batch tokenization and model inference
        all_logits = []
        total_pairs = len(pairs)
        
        for batch_start in range(0, total_pairs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            # Tokenize batch - properly handle sentence pairs
            inputs = self.tokenizer(
                [sent for sent, _ in batch_pairs],
                [sample for _, sample in batch_pairs],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass - keep logits for E/C renormalization
            logits = self.model(**inputs).logits
            all_logits.append(logits.cpu())
        
        # Concatenate all logits (ensure on CPU)
        if len(all_logits) == 0:
            # Edge case: no pairs (shouldn't happen due to check above, but be safe)
            entailment_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
            neutral_scores = None  # Always None, neutral is ignored
            contradiction_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
            return entailment_scores, neutral_scores, contradiction_scores
        
        all_logits = torch.cat(all_logits, dim=0)  # shape: (total_pairs, num_labels)
        # Ensure all tensors are on CPU before converting to numpy
        all_logits = all_logits.cpu()
        
        # Convert pair_indices to numpy arrays for vectorized indexing
        sent_indices = np.array([sent_i for sent_i, _ in pair_indices])
        sample_indices = np.array([sample_i for _, sample_i in pair_indices])
        
        # Initialize output matrices
        entailment_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
        neutral_scores = None  # Always None, neutral is ignored
        contradiction_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
        
        # Always use E/C renormalization (drop neutral) - matching original paper
        # Extract E and C logits for all pairs at once
        ec_logits = all_logits[:, [self.entailment_idx, self.contradiction_idx]]  # (total_pairs, 2)
        ec_probs = torch.softmax(ec_logits, dim=1)  # (total_pairs, 2)
        
        # Vectorized assignment (tensors are on CPU, safe to use .numpy())
        entailment_scores[sent_indices, sample_indices] = ec_probs[:, 0].cpu().numpy()
        contradiction_scores[sent_indices, sample_indices] = ec_probs[:, 1].cpu().numpy()
        
        return entailment_scores, neutral_scores, contradiction_scores

    def extract_features(self, sentences, sampled_passages):
        """
        Extract contradiction-derived features for supervised learning.
        Features: mean(C), max(C), std(C), frac(C>0.5), entropy(C), iqr(C), top3_mean(C).
        Neutral is ignored (always uses E/C renormalization).
        :param sentences: list[str] -- sentences to be evaluated
        :param sampled_passages: list[str] -- stochastically generated responses
        :return features: np.array of shape (num_sentences, 7) - 7 features per sentence
        """
        # Call predict_matrix once for all sentences (batched and efficient)
        entailment_scores, neutral_scores, contradiction_scores = self.predict_matrix(
            sentences, sampled_passages
        )
        
        features = []
        for sent_i in range(len(sentences)):
            contras = contradiction_scores[sent_i, :]
            
            # Basic contradiction statistics
            contra_feat = [
                np.mean(contras),      # mean(C)
                np.max(contras),       # max(C)
                np.std(contras),       # std(C)
                np.mean(contras > 0.5), # frac(C > 0.5)
            ]
            
            # Entropy computed from [C, 1-C] distribution
            if len(contras) > 0:
                # Normalize to ensure valid probability distribution
                c_probs = np.clip(contras, 1e-10, 1.0 - 1e-10)  # Avoid log(0)
                not_c_probs = 1.0 - c_probs
                probs_matrix = np.stack([c_probs, not_c_probs], axis=0)  # (2, num_samples)
                # Compute entropy for each sample: -sum(p * log(p))
                entropies = -np.sum(probs_matrix * np.log(probs_matrix + 1e-10), axis=0)
                entropy_feat = [np.mean(entropies)]
            else:
                entropy_feat = [0.0]
            
            # IQR of contradiction scores
            if len(contras) > 0:
                iqr_c = np.percentile(contras, 75) - np.percentile(contras, 25)
            else:
                iqr_c = 0.0
            iqr_feat = [iqr_c]
            
            # Top-k: mean of top 3 contradiction probs
            if len(contras) > 0:
                sorted_contras = np.sort(contras)[::-1]  # descending order
                top_k = min(3, len(sorted_contras))
                topk_feat = [np.mean(sorted_contras[:top_k])]
            else:
                topk_feat = [0.0]
            
            # Combine all features (only contradiction-derived)
            feat = contra_feat + entropy_feat + iqr_feat + topk_feat
            features.append(feat)
        
        return np.array(features)


class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device = None
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False, # hf's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
