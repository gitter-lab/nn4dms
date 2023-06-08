""" Some useful constants used throughout codebase """

DATASETS = {"avgfp": {"ds_name": "avgfp",
                      "ds_dir": "data/avgfp",
                      "ds_fn": "data/avgfp/avgfp.tsv",
                      "wt_aa": "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
                               "VTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTL"
                               "VNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQL"
                               "ADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                      "wt_ofs": 0,
                      "pdb_fn": "data/avgfp/avgfp_rosetta_model.pdb"},

            "avgfp_metl": {"ds_name": "avgfp_metl",
                           "ds_dir": "data/avgfp_metl",
                           "ds_fn": "data/avgfp_metl/avgfp.tsv",
                           "wt_aa": "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
                                    "VTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTL"
                                    "VNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQL"
                                    "ADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                           "wt_ofs": 0,
                           "pdb_fn": "data/avgfp/avgfp_rosetta_model.pdb"},

            "bgl3": {"ds_name": "bgl3",
                     "ds_dir": "data/bgl3",
                     "ds_fn": "data/bgl3/bgl3.tsv",
                     "wt_aa": "MVPAAQQTAMAPDAALTFPEGFLWGSATASYQIEGAAAEDGRTPSIWDTYARTPGRVRNGDTGDVAT"
                              "DHYHRWREDVALMAELGLGAYRFSLAWPRIQPTGRGPALQKGLDFYRRLADELLAKGIQPVATLYHWD"
                              "LPQELENAGGWPERATAERFAEYAAIAADALGDRVKTWTTLNEPWCSAFLGYGSGVHAPGRTDPVAALR"
                              "AAHHLNLGHGLAVQALRDRLPADAQCSVTLNIHHVRPLTDSDADADAVRRIDALANRVFTGPMLQGAYP"
                              "EDLVKDTAGLTDWSFVRDGDLRLAHQKLDFLGVNYYSPTLVSEADGSGTHNSDGHGRSAHSPWPGADRVA"
                              "FHQPPGETTAMGWAVDPSGLYELLRRLSSDFPALPLVITENGAAFHDYADPEGNVNDPERIAYVRDHLAAV"
                              "HRAIKDGSDVRGYFLWSLLDNFEWAHGYSKRFGAVYVDYPTGTRIPKASARWYAEVARTGVLPTAGDPNSSS"
                              "VDKLAAALEHHHHHH",
                     "wt_ofs": 0,
                     "pdb_fn": "data/bgl3/bgl3_rosetta_model.pdb"},

            "gb1": {"ds_name": "gb1",
                    "ds_dir": "data/gb1",
                    "ds_fn": "data/gb1/gb1.tsv",
                    "wt_aa": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                    "wt_ofs": 0,
                    "pdb_fn": "data/gb1/2qmt.pdb"},

            "pab1": {"ds_name": "pab1",
                     "ds_dir": "data/pab1",
                     "ds_fn": "data/pab1/pab1.tsv",
                     "wt_aa": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP",
                     "wt_ofs": 126,
                     "pdb_fn": "data/pab1/pab1_rosetta_model.pdb"},

            "ube4b": {"ds_name": "ube4b",
                      "ds_dir": "data/ube4b",
                      "ds_fn": "data/ube4b/ube4b.tsv",
                      "wt_aa": "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLP"
                               "SGTVMDRSIILRHLLNSPTDPFNRQMLTESMLEPVPELKEQIQAWMREKQSSDH",
                      "wt_ofs": 0,
                      "pdb_fn": "data/ube4b/ube4b_rosetta_model.pdb"}}


# list of chars that can be encountered in any sequence
CHARS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# number of chars
NUM_CHARS = 21  # len(CHARS)

# dictionary mapping chars->int
C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}
