# Source data
This directory contains the raw source data for each dataset, acquired from supplemental information sections or directly from authors.
It also contains the source AAindex database.

| Dataset | Reference                                                                                         | First Author | Year | Acquired From                                                       | Link |
|---------|---------------------------------------------------------------------------------------------------|--------------|------|---------------------------------------------------------------------|------|
| avgfp   | Local fitness landscape of the green fluorescent protein                                          | Sarkisyan    | 2016 | Associated data on figshare, amino_acid_genotypes_to_brightness.tsv | [Link](http://dx.doi.org/10.6084/m9.figshare.3102154) |
| bgl3    | Dissecting enzyme function with microfluidic-based deep mutational scanning                       | Romero       | 2015 | Sequence read archive                                                  | [Link](https://www.ncbi.nlm.nih.gov/bioproject?LinkName=sra_bioproject&from_uid=10490386) |
| gb1     | A comprehensive biophysical description of pairwise epistasis throughout an entire protein domain | Olson        | 2014 | Supplemental information, Table S2                                  | [Link](https://www.cell.com/current-biology/fulltext/S0960-9822(14)01268-8#supplementaryMaterial) |
| pab1    | Deep mutational scanning of an RRM domain of the Saccharomyces cerevisiae poly(A)-binding protein | Melamed      | 2013 | Supplemental material, Supp Table 2 and Supp Table 5                | [Link](https://rnajournal.cshlp.org/content/suppl/2013/09/09/rna.040709.113.DC1.html) |
| ube4b   | Activity-enhancing mutations in an E3 ubiquitin ligase identified by high-throughput mutagenesis  | Starita      | 2013 | Supporting Information, Dataset_S01, nscor_log2_ratio               | [Link](https://www.pnas.org/content/suppl/2013/03/15/1303309110.DCSupplemental) |

AAIndex database: https://www.genome.jp/aaindex/


We processed this raw data into a uniform format that can be used to train models with our codebase. The processed data is contained in the [data](../data) directory. The scripts we used to [process the data](../code/parse_source_data.py) and the [AAIndex database](../code/parse_aaindex.py) are provided for reference.

For Bgl3, we used [bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) to align raw sequencing reads (linked above) and computed variant counts based on the resulting sequence alignment maps. The Bgl3 source data directory contains text files with all the variant reads in both the "unlabeled" set (initial library) and the "positive" set (post function selection). 