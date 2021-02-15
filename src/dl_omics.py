from urllib.request import urlretrieve
import os

from cmapPy.pandasGEXpress import parse
from pandas import read_excel, read_csv
from joblib import dump, load

from utils import gunzip

current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_file_path, os.pardir, 'data')

meta_data_url = 'https://ndownloader.figstatic.com/files/25612817'
meta_data_file = os.path.join(data_path, 'Table_1_meta_data.xlsx')

raw_counts_url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file' \
                 '=GSE92742%5FBroad%5FLINCS%5FLevel5%5FCOMPZ%2EMODZ%5Fn473647x12328%2Egctx%2Egz'

raw_counts_file_compressed = os.path.join(
    data_path, 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz')
raw_counts_file = raw_counts_file_compressed[:-3]

gene_info_url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE92742&format=file&file' \
                '=GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo%2Etxt%2Egz'
gene_info_file = os.path.join(data_path, 'GSE92742_Broad_LINCS_gene_info.txt.gz')

l1000_df_file = os.path.join(data_path, 'l1000_df.bin')

N_GENES = 12_328


def create_l1000_df():

    os.makedirs(data_path, exist_ok=True)

    if os.path.exists(l1000_df_file):
        return load(l1000_df_file)

    if not os.path.exists(raw_counts_file):

        if not os.path.exists(raw_counts_file_compressed):
            print('Downloading counts from GEO...')
            urlretrieve(raw_counts_url, filename=raw_counts_file_compressed)

        gunzip(raw_counts_file_compressed, raw_counts_file)

    if not os.path.exists(gene_info_file):
        urlretrieve(gene_info_url, filename=gene_info_file)

    if not os.path.exists(meta_data_file):
        urlretrieve(meta_data_url, filename=meta_data_file)

    meta_df = read_excel(meta_data_file)
    sample_names = meta_df['NIH LINCS L1000 sig_id']

    gene_df = read_csv(gene_info_file, sep='\t')
    assert gene_df.shape[0] == N_GENES
    gene_df = gene_df.set_index('pr_gene_id')

    counts_df = parse.parse(raw_counts_file).data_df
    assert counts_df.shape[0] == N_GENES
    counts_df = counts_df[sample_names]
    counts_df.index = counts_df.index.astype(int)

    df = counts_df.join(gene_df, how='inner')
    assert df.shape[0] == N_GENES
    df = df.loc[df.pr_is_lm == 1, :]
    df = df.set_index('pr_gene_symbol')
    assert df.index.is_unique
    df = df[sample_names].T

    columns_in_excel_table = [
        'NIH LINCS L1000 sig_id',
        'Usage',
        'Compound Names',
        'DILIst binaray classfication ',
        'Cell_id',
        'Dose',
        'Treatment Duration (hr)']
    meta_df = meta_df[columns_in_excel_table]
    meta_df.columns = [
        'sample_id',
        'set',
        'compound',
        'dili',
        'cell_id',
        'dose',
        'duration'
    ]
    meta_df = meta_df.set_index('sample_id')
    meta_columns = meta_df.columns

    genes = df.columns.to_list()
    df = df.join(meta_df)

    dump((df, genes, meta_columns), l1000_df_file, compress=3)

    return df, genes, meta_columns


def main():

    df, genes, meta_columns = create_l1000_df()


if __name__ == '__main__':
    main()
