# Creating Deep Learning Models for Classification of Sequencing Artifacts in Long-Read Whole Genome Sequencing (WGS) Data

The goal of this project is to build deep learning classification models to distinguish artifactual variant calls from genuine artifacts in long-read sequencing data. The three major steps (corresponding to the three notebooks here) for carrying this out are the following:

1. [Processing raw FAST5 files to variant calls](https://github.com/felixm3/deep-learning-long-read-sequencing/blob/main/01_fastq_to_vcf.ipynb) which includes:
   - Downloading raw FAST5 data from the Nanopore Whole Genome Sequencing Consortium GitHub repository,
   - base calling to get FASTQ files,
   - mapping to the human reference genome to get BAM files,
   - variant calling to get VCF files, and
   - intersecting with NIST Genome-in-a-Bottle gold-standard benchmarking data to allow determination of which variant calls were correct vs artifacts.

2. [Preprocessing of the VCFs to a format usable for deep learning modeling](https://github.com/felixm3/deep-learning-long-read-sequencing/blob/main/02_preprocessing_vcfs_for_deep_learning.ipynb) which includes:
   - extracting sequence context of all variants in the VCF files from the reference genome
   - creating a Pandas features-labels dataframe with the sequence context being the feature and artifact/not-artifact being the label
   - splitting the data into train-validation and test
   - encoding the sequences and labels into Numpy arrays for input into the deep learning models
  
3. [Fitting and evaluating various deep learning models](https://github.com/felixm3/deep-learning-long-read-sequencing/blob/main/03_deep_learning_modeling_for_sequence_classification.ipynb) including:
   - multilayer perceptrons (MLPs)
   - convolutional neural networks (CNNs)
   - recurrent neural networks (RNNs) e.g. LSTMs
