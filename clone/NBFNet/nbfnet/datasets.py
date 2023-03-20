import os

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import re  # MODIFY


class IndRelLinkPredDataset(InMemoryDataset):
    urls = {
        # \\:"FB15k-237": [
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        # \\:],
        # \\:"WN18RR": [
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        # \\:    "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        # \\:],
        "FD1Trans": [],  # MODIFY
        "FD1Ind": [],  # MODIFY
        "FD2Trans": [],  # MODIFY
        "FD2Ind": [],  # MODIFY
        "WN18RR1Trans": [],  # MODIFY
        "WN18RR1Ind": [],  # MODIFY
        "WN18RR1PermInd": [],  # MODIFY
        "WN18RR1SwapInd": [],  # MODIFY
        "NELL9951Trans": [],  # MODIFY
        "NELL9951Ind": [],  # MODIFY
        "NELL9951PermInd": [],  # MODIFY
        "NELL9951SwapInd": [],  # MODIFY
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        # \\:self.version = version
        # \\:assert name in ["FB15k-237", "WN18RR"]
        # \\:assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        # \\:return int(self.data.edge_type.max()) + 1
        inv_relation_vocab = {}  # MODIFY
        with open(  # MODIFY
            os.path.join(self.raw_dir, "relations.dict"),  # MODIFY
            "r",  # MODIFY
        ) as file:  # MODIFY
            # # MODIFY
            for line in file:  # MODIFY
                # # MODIFY
                msg_id, msg_name = re.split(r"\s+", line.strip())  # MODIFY
                inv_relation_vocab[msg_name] = int(msg_id)  # MODIFY
        num_relations = len(inv_relation_vocab) * 2  # MODIFY
        return num_relations  # MODIFY

    @property
    def raw_dir(self):
        # \\:return os.path.join(self.root, self.name, self.version, "raw")
        return os.path.join(self.root, self.name, "raw")  # MODIFY

    @property
    def processed_dir(self):
        # \\:return os.path.join(self.root, self.name, self.version, "processed")
        return os.path.join(self.root, self.name, "processed")  # MODIFY

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        # \\:return ["train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"]
        return ["test.txt", "valid.txt", "train.txt", "observe.txt"]  # MODIFY

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]
        is_trans = self.name[-5:] == "Trans"  # MODIFY
        is_ind = self.name[-3:] == "Ind"  # MODIFY
        assert is_trans or is_ind  # MODIFY
        if is_trans:  # MODIFY
            title = self.name[:-5]  # MODIFY
            assert title + "Trans" == self.name  # MODIFY
            test_files = [test_files[1]]  # MODIFY
            if title in ["FD1", "FD2"]:  # MODIFY
                # # MODIFY
                train_files = [train_files[0], train_files[1]]  # MODIFY
            else:  # MODIFY
                # # MODIFY
                train_files = [train_files[0]]  # MODIFY
        elif is_ind:  # MODIFY
            title = self.name[:-3]  # MODIFY
            assert title + "Ind" == self.name  # MODIFY
            test_files = [test_files[0]]  # MODIFY
            train_files = [train_files[1]]  # MODIFY

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        with open(  # MODIFY
            os.path.join(os.path.dirname(test_files[0]), "entities.dict"), "r"  # MODIFY
        ) as file:  # MODIFY
            # # MODIFY
            for line in file:  # MODIFY
                # # MODIFY
                msg_id, msg_name = re.split(r"\s+", line.strip())  # MODIFY
                inv_train_entity_vocab[msg_name] = int(msg_id)  # MODIFY
                inv_test_entity_vocab[msg_name] = int(msg_id)  # MODIFY
        with open(  # MODIFY
            os.path.join(os.path.dirname(test_files[0]), "relations.dict"),
            "r",  # MODIFY
        ) as file:  # MODIFY
            # # MODIFY
            for line in file:  # MODIFY
                # # MODIFY
                msg_id, msg_name = re.split(r"\s+", line.strip())  # MODIFY
                inv_relation_vocab[msg_name] = int(msg_id)  # MODIFY
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    # \\:h_token, r_token, t_token = line.strip().split("\t")
                    h_token, r_token, t_token = re.split(r"\s+", line.strip())  # MODIFY
                    # \\:if h_token not in inv_train_entity_vocab:
                    # \\:    inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    # \\:if r_token not in inv_relation_vocab:
                    # \\:    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    # \\:if t_token not in inv_train_entity_vocab:
                    # \\:    inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    # \\:h_token, r_token, t_token = line.strip().split("\t")
                    h_token, r_token, t_token = re.split(r"\s+", line.strip())  # MODIFY
                    # \\:if h_token not in inv_test_entity_vocab:
                    # \\:    inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    # \\:assert r_token in inv_relation_vocab
                    # \\:if r_token not in inv_relation_vocab:  # MODIFY
                    # \\:    inv_relation_vocab[r_token] = len(inv_relation_vocab)  # MODIFY
                    r = inv_relation_vocab[r_token]
                    # \\:if t_token not in inv_test_entity_vocab:
                    # \\:    inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        assert set(inv_test_entity_vocab.keys()).issubset(
            set(inv_train_entity_vocab.keys()),
        )  # MODIFY
        num_entities = len(inv_train_entity_vocab)  # MODIFY

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # \\:train_fact_slice = slice(None, sum(num_samples[:1]))
        # \\:test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_slice = slice(None, sum(num_samples[: len(train_files)]))
        test_fact_slice = slice(None, sum(num_samples[: len(train_files)]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        # \\:train_slice = slice(None, sum(num_samples[:1]))
        # \\:valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # \\:test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_slice = slice(None, sum(num_samples[: len(train_files)]))  # MODIFY
        valid_slice = slice(sum(num_samples[: len(train_files)]), sum(num_samples))  # MODIFY
        test_slice = slice(sum(num_samples[: len(train_files)]), sum(num_samples))  # MODIFY
        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            # \\:num_nodes=len(inv_train_entity_vocab),
            num_nodes=num_entities,  # MODIFY
            target_edge_index=edge_index[:, train_slice],
            target_edge_type=edge_type[train_slice],
        )
        valid_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            # \\:num_nodes=len(inv_train_entity_vocab),
            num_nodes=num_entities,  # MODIFY
            target_edge_index=edge_index[:, valid_slice],
            target_edge_type=edge_type[valid_slice],
        )
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            # \\:num_nodes=len(inv_test_entity_vocab),
            num_nodes=num_entities,  # MODIFY
            target_edge_index=edge_index[:, test_slice],
            target_edge_type=edge_type[test_slice],
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        assert train_data.num_nodes == valid_data.num_nodes and train_data.num_nodes == test_data.num_nodes  # MODIFY
        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name
