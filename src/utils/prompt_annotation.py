from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import pairwise_distances
import argparse
import os
import torch
import pickle


def load_raw_sentences(args, domain, mode):
    data_dir = os.path.join('data', args.root_data_dir, domain)
    train_path = os.path.join(data_dir, mode)
    print("LOOKING AT {}".format(train_path))
    with open(train_path, 'rb') as f:
        (sentences, labels) = pickle.load(f)
    if isinstance(sentences[0], list):
        sentences = [' '.join(snt) for snt in sentences]
    return sentences, labels


def load_data_and_encode(args, domain, mode, model, tokenizer):
    raw_sentences, labels = load_raw_sentences(args, domain, mode)
    sentences, embeddings = [], []
    for snt in raw_sentences:
        if type(snt) is tuple:
            total_snt = snt[0] + " " + snt[1]
        else:
            total_snt = snt
        sentences.append(total_snt)
        with torch.no_grad():
            cur_emb = model.get_input_embeddings()(tokenizer(total_snt, return_tensors='pt')['input_ids'][:, :-1]).squeeze()
            embeddings.append(cur_emb)
    return sentences, embeddings, labels


def return_unique_list_and_keep_sorted(sorted_closets_drfs_ids):
    unique_list = []
    for id in sorted_closets_drfs_ids:
        if id not in unique_list:
            unique_list.append(id)
    return unique_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--domains",
                        default='charliehebdo,ferguson,germanwings-crash,ottawashooting,sydneysiege',
                        type=str,
                        required=False,
                        help="The domain names separated with a comma - NO SPACES.")
    parser.add_argument("--root_data_dir",
                        choices=['rumor_data', 'mnli_data', 'absa_data'],
                        default='rumor_data',
                        type=str,
                        required=False,
                        help="The root data directory - 'absa_data', 'mnli_data', or 'rumour_data'.")
    parser.add_argument("--k_neighbours",
                        default=5,
                        type=int,
                        required=False,
                        help="Number of nearest DRFs per example.")
    parser.add_argument("--drf_set_location",
                        default='./runs/rumor/charliehebdo/drf_sets',
                        type=str,
                        required=True,
                        help="Location of extracted drf sets.")
    parser.add_argument("--prompts_data_dir",
                        default='./runs/rumor/charliehebdo/prompt_annotations',
                        type=str,
                        required=True,
                        help="Location to save annotated prompts.")
    args = parser.parse_args()
    domains_list = args.domains.split(',')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    model.eval()
    modes = ['train']
    for mode in modes:
        for domain in domains_list:
            trg_sentences, trg_embeddings, _ = load_data_and_encode(args, domain, mode, model, tokenizer)

            with open(os.path.join(args.drf_set_location, f'{domain}.pkl'), 'rb') as f:
                drfs = pickle.load(f)

            drfs_emb = []
            with torch.no_grad():
                for drf in drfs:
                    drf_ids = tokenizer(drf, return_tensors='pt')['input_ids'][:, :-1]
                    drf_emb = model.get_input_embeddings()(drf_ids).mean(dim=1).squeeze()
                    drfs_emb.append(drf_emb)
            drfs_emb = torch.stack(drfs_emb).numpy()

            nearest_drfs_per_example, closest_distances_per_example = [], []
            for sent_id, sentence_emb in enumerate(trg_embeddings):
                if len(sentence_emb.shape) == 1:
                    sentence_emb = sentence_emb.unsqueeze(axis=0)
                pairwise_dist = pairwise_distances(sentence_emb, drfs_emb, metric='minkowski')
                per_drf_min_dist = pairwise_dist.min(axis=0)
                sorted_drf_min_distances = sorted(range(len(per_drf_min_dist)), key=per_drf_min_dist.__getitem__)
                closest_drf_ids = sorted_drf_min_distances[:args.k_neighbours]
                closest_drfs = [drfs[piv_id] for piv_id in closest_drf_ids]

                prompt_string = ', '.join(closest_drfs)
                nearest_drfs_per_example.append(closest_drfs)
                if sent_id < 1:
                    print("------------- Prompt Annotation Example -------------")
                    print(f"Sentence: {trg_sentences[sent_id]}")
                    print(f"Prompt: {prompt_string}")
                    print("-----------------------------------------------------")

            if not os.path.exists(os.path.join(args.prompts_data_dir, domain)):
                os.makedirs(os.path.join(args.prompts_data_dir, domain))
            file_name = os.path.join(args.prompts_data_dir, domain, 'annotated_prompts_train.pt')
            torch.save(nearest_drfs_per_example, file_name)


if __name__ == "__main__":
    main()

