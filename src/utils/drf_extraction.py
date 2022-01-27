from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import argparse
import pickle
import os


def GetTopNMI(n, X, target):
    MI = []
    length = X.shape[1]

    for i in range(length):
        temp = mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs, MI


def getCounts(X,i):
    return (sum(X[:,i]))


def preproc(args):
    """find DRFs from source and data domains with mutual information
    Parameters in args:
    drf_num (list of int or int): number of DRFs to find
    drf_min_st (int): minimal appearances of the DRFs in both source and non_source domains
    n_gram (tuple of integers): n_grams to include in DRFs selection (min, max), default is 1 grams only
    drf_set_location (str): Location for saving the extracted drf sets
    domains (List[str]]: names of source domains
    dtype (str): data name, task name
    Returns:
    list of DRFs
   """
    domains = args.domains.split(',')
    data_type = args.dtype  # 'rumour_data'

    for src in domains:
        if os.path.exists(os.path.join(args.drf_set_location, f'{src}.pkl')):
            print(f"The following file already exists: {os.path.join(args.drf_set_location, f'{src}.pkl')}. "
                  f"Overwritng!!!")
        train, train_labels, non_source = [], [], []

        # Accumulate train examples - examples from all available domains.
        # Label src examples as '1' and non-source examples as '0'.
        # This will be used for the Mutual Information calculation between N-grams and the domain label.
        for domain in domains:
            dom_id = 1 if domain == src else 0
            train_path = os.path.join("data", f'{data_type}_data', domain, "train")
            with open(train_path, 'rb') as f:
                (tmp_train, _) = pickle.load(f)
                tmp_domain_labels = [dom_id] * len(tmp_train)
            train_labels = train_labels + tmp_domain_labels
            if data_type == 'mnli':
                for tr_ex in tmp_train:
                    train.append(tr_ex[0] + " " + tr_ex[1])
            else:
                train = train + tmp_train

        # Accumulate only source examples.
        # This will be used for measuring N-grams frequency (appearance) in the source domain.
        src_path = os.path.join("data", f'{data_type}_data', src, "train")
        source = []
        with open(src_path, 'rb') as f:
            (tmp_source, _) = pickle.load(f)
        if data_type == 'mnli':
            for src_ex in tmp_source:
                source.append(src_ex[0] + " " + src_ex[1])
        else:
            source = tmp_source

        # Accumulate only non-source examples.
        # This will be used for measuring N-grams frequency (appearance) in the non-source domains.
        for domain in domains:
            if domain != src:
                tmp_non_src_path = os.path.join("data", f'{data_type}_data', domain, "train")
                with open(tmp_non_src_path, 'rb') as f:
                    (tmp_non_src, _) = pickle.load(f)
                if data_type == 'mnli':
                    for trg_ex in tmp_non_src:
                        non_source.append(trg_ex[0] + " " + trg_ex[1])
                else:
                    non_source = non_source + tmp_non_src

        if isinstance(train[0], list):
            train = [' '.join(train_instance) for train_instance in train]
            source = [' '.join(src_instance) for src_instance in source]
            non_source = [' '.join(trg_instance) for trg_instance in non_source]

        # Calc train counts
        bigram_vectorizer_train = CountVectorizer(ngram_range=args.n_gram, token_pattern=r'\b\w+\b', min_df=5,
                                                  binary=True, stop_words='english')
        counts_train = bigram_vectorizer_train.fit_transform(train).toarray()

        # Calc source counts
        bigram_vectorizer_source = CountVectorizer(ngram_range=args.n_gram, token_pattern=r'\b\w+\b',
                                                   min_df=args.drf_min_st, binary=True, stop_words='english')
        counts_source = bigram_vectorizer_source.fit_transform(source).toarray()

        # Calc non-source counts
        bigram_vectorizer_non_source = CountVectorizer(ngram_range=args.n_gram, token_pattern=r'\b\w+\b',
                                                   min_df=args.drf_min_st, binary=True, stop_words='english')
        counts_non_source = bigram_vectorizer_non_source.fit_transform(non_source).toarray()

        # get a sorted list of DRFs with respect to the MI with the label
        MIsorted, RMI = GetTopNMI(100, counts_train, train_labels)
        MIsorted.reverse()
        c = 0

        drf_names = []
        for i, MI_word in enumerate(MIsorted):
            name = bigram_vectorizer_train.get_feature_names()[MI_word]
            if len(name) < 3 or name.isnumeric() or (len(name) == 2 and any(char.isdigit() for char in name)):
                continue

            s_count = getCounts(counts_source, bigram_vectorizer_source.get_feature_names().index(
                name)) if name in bigram_vectorizer_source.get_feature_names() else 0
            t_count = getCounts(counts_non_source, bigram_vectorizer_non_source.get_feature_names().index(
                name)) if name in bigram_vectorizer_non_source.get_feature_names() else 0

            if s_count > 0 and float(t_count) / s_count <= args.rho:
                drf_names.append(name)
                c += 1

            if c >= args.drf_num:
                break

        # print("Number of chosen DRFs:", len(drf_names))
        # print("DRFs:", drf_names[:args.drf_num])
        if not os.path.exists(args.drf_set_location):
            os.makedirs(args.drf_set_location)
        filename = os.path.join(args.drf_set_location, f'{src}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(drf_names[:args.drf_num], f)
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--domains",
                        default='ferguson,germanwings-crash,ottawashooting,sydneysiege',
                        type=str,
                        required=True,
                        help="The domain names separated with a comma - NO SPACES.")
    parser.add_argument("--dtype",
                        choices=['absa', 'rumor', 'mnli'],
                        default='rumor',
                        type=str,
                        required=True,
                        help="The task name.")
    parser.add_argument("--drf_num",
                        default=1000,
                        type=int,
                        help="The number of selected DRFs")
    parser.add_argument("--drf_min_st",
                        default=20,
                        type=int,
                        help="Minimum counts of DRFs in src and in dest")
    parser.add_argument("--n_gram",
                        default='unigram',
                        type=str,
                        help="N_gram length.")
    parser.add_argument("--drf_set_location",
                        default='./runs/rumor/charliehebdo/drf_sets',
                        type=str,
                        required=True,
                        help="Location of extracted drf sets.")
    parser.add_argument("--rho",
                        default=1.5,
                        type=float,
                        required=False,
                        help="Rho parameter value")

    args = parser.parse_args()

    if args.n_gram == "bigram":
        args.n_gram = (1, 2)
    elif args.n_gram == "unigram":
        args.n_gram = (1, 1)
    else:
        print("This code does not soppurt this type of n_gram")
        exit(-1)

    preproc(args)


if __name__ == "__main__":
    main()
