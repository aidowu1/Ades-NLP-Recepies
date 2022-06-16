import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple

class Helpers(object):
    """
    Helper utility functions
    """

    @staticmethod
    def tableize(df):
        """
        Tabulate the dataframe
        :param df: Input dataframe
        :return: Output pretty dispaly of dataframe
        """
        if not isinstance(df, pd.DataFrame):
            return
        df_columns = df.columns.tolist()
        max_len_in_lst = lambda lst: len(sorted(lst, reverse=True, key=len)[0])
        align_center = lambda st, sz: "{0}{1}{0}".format(" " * (1 + (sz - len(st)) // 2), st)[:sz] if len(
            st) < sz else st
        align_right = lambda st, sz: "{0}{1} ".format(" " * (sz - len(st) - 1), st) if len(st) < sz else st
        max_col_len = max_len_in_lst(df_columns)
        max_val_len_for_col = dict(
            [(col, max_len_in_lst(df.iloc[:, idx].astype('str'))) for idx, col in enumerate(df_columns)])
        col_sizes = dict([(col, 2 + max(max_val_len_for_col.get(col, 0), max_col_len)) for col in df_columns])
        build_hline = lambda row: '+'.join(['-' * col_sizes[col] for col in row]).join(['+', '+'])
        build_data = lambda row, align: "|".join(
            [align(str(val), col_sizes[df_columns[idx]]) for idx, val in enumerate(row)]).join(['|', '|'])
        hline = build_hline(df_columns)
        out = [hline, build_data(df_columns, align_center), hline]
        for _, row in df.iterrows():
            out.append(build_data(row.tolist(), align_right))
        out.append(hline)
        return "\n".join(out)

    def createLowerTriangularMatrixOfPairs(n_sentences: int):
        """
        Create triangular matrix indices doc_pair_indices for the similarity measure
        """
        matrix = np.zeros((n_sentences, n_sentences))
        indices = np.tril_indices_from(matrix)
        n_rows = indices[0].shape[0]
        pairs = [(indices[0][i], indices[1][i]) for i in range(n_rows)
                 if not indices[0][i] == indices[1][i]]
        return pairs

    @staticmethod
    def sliceDict(map: Dict, n_samples: int):
        """
        Slice the dictionary to a specified number of samples
        :param map: Input dictionary
        :param n_samples: Number of samples
        :return: Sliced dictionary
        """
        return dict(itertools.islice(map.items(), n_samples))

    def displaySortedSimilarityMeasures(
            docs: List[str],
            similarity_result: Dict[str, float],
            top_k: int = None) -> pd.DataFrame:
        """
        Display computed top k similarities
        :param docs: Documents/sentences
        :param top_k: Number of top k similarities
        :param similarity_result: Similarity results
        :return: Results as a table
        """
        query_doc_ids = []
        current_doc_ids = []
        query_doc = []
        current_doc = []
        scores = []
        print(f"Total number of computed similarity scores: {len(similarity_result.items())}")
        print(f"\nSorted Similarity Measures (in descending) order and top {top_k} are:\n\n")
        sorted_sims = sorted(similarity_result.items(), key=lambda item: item[1], reverse=True)
        if top_k:
            sorted_sims = sorted_sims[:top_k]
        for key, value in sorted_sims:
            doc_ids = tuple([int(x) for x in key.split("_")])
            doc_1_id, doc_2_id = doc_ids
            query_doc_ids.append(doc_1_id)
            current_doc_ids.append(doc_2_id)
            query_doc.append(docs[doc_1_id])
            current_doc.append(docs[doc_2_id])
            scores.append(value)
        result_df = pd.DataFrame(data={
            "query_doc_id": query_doc_ids,
            "current_doc_id": current_doc_ids,
            "query_doc": query_doc,
            "current_doc": current_doc,
            "similarity score": scores
        })
        return result_df

