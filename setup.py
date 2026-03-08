from setuptools import setup

# reading long description from file
with open('DESCRIPTION.txt') as file:
    long_description = file.read()

REQUIREMENTS = [
    "numpy>=1.21,<2",
    "torch>=1.12",
    "transformers>=4.35,<4.45",  # 4.45+ requires torch>=2.4 (not on all platforms)
    "sentencepiece",
    "protobuf",
    "bert_score",
    "spacy",
    "nltk",
    "openai",
    "groq",
    "datasets",
    "scikit-learn",
    "matplotlib",
    "tqdm",
]

# some more details
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

VERSION = {}
with open("selfcheckgpt/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# calling the setup function
setup(
    name='selfcheckgpt',
    version=VERSION["__version__"],
    description='SelfCheckGPT: Assessing text-based responses from LLMs',
    long_description=long_description,
    url='https://github.com/potsawee/selfcheckgpt',
    author='Potsawee Manakul',
    author_email='m.potsawee@gmail.com',
    license='MIT',
    packages=['selfcheckgpt'],
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords='selfcheckgpt',
    include_package_data=True,
)
