# PhosTransfer

Requirements:

- python 2.7
- tensorflow
- Numpy
- matplotlib
- shutil
- sklearn
- PsiBlast
- PSIPRED (optional)
- DISOPRED3 (optional)


## Preprocessing: feature extraction

Config #home_path, #uniref90_psi_blast_database, #blast_path, #tool_dir in src/prep_vec/util.py: 

- home_path: path to saving generated feature files. 
- uniref90_psi_blast_database: path to blast database. We use the <a href="ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz">uniref90filt</a> database. 
- blast_path: path to exculable psiblast. Install from <a href="https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download">PsiBlast</a>. 
- tool_dir: path to feature generation tools. Install from <a href="http://bioinf.cs.ucl.ac.uk/software_downloads/">PSIPRED</a> and <a href="http://bioinf.cs.ucl.ac.uk/software_downloads/">DISOPRED3</a>. 

Pre-genearted features can be downloaded <a href="https://drive.google.com/open?id=1j8CVtFLiYmHLjEuUcJ5b5Y0WszlP7CSk">here</a>. 
To regenerate features from scratch, please configure and run src/prep_vec/residue2vec.py. 

## Model training and testing

Config and run src/phospho_prediction.py. 

- mode=cv: run cross validation. Trained models will be saved to OUTPUT/checkpoints and the best models will be saved to FINALS/models. 
- mode=test: run independent test. Test performance will be saved to #output_log that is configured. 

Pretrained models for different kinases can be downloaded <a href="https://drive.google.com/open?id=12jqviDdJ_GonVboTCSr526AGRoRgD3Ew">here</a>. 
The predicted results for our independent tests can be found in FINALS/predicts. 

## Benchmark dataset

Here we release the benchmark dataset for hierarchical phosphorylation site prediction. The DATA directory structure is as follows

- Combined_train
   - ST 
      - sites: annotated phosphorylation sites
      - chains: amino acid sequences in Fasta format
      - ID.txt: protein identifications
      - sub-directories: children nodes
   - T (same as above)
- Combined_test
   - ST (same as above)
   - T (same as above)
   
The benchmark dataset can be downloaded <a href="https://drive.google.com/open?id=17MR0EweH8jtbIB35TistFnWir2bOXR3_">here</a>. 