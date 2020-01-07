import torch

from torchreid.data.datamanager import ImageDataManager



def build_data_loader(data_dir, domain_datasets, mode, num_workers, num_instances=1, batch_id_size=32, batch_size=32):
    if mode == "train":
        datamanager = ImageDataManager(
            root=data_dir,
            sources=domain_datasets,
            targets=[],
            combineall=True,
            transforms=["random_flip", "random_crop"],
            train_sampler="RandomIdentitySampler",
            num_instances=num_instances,
            batch_size_train=batch_id_size * num_instances,
            workers=num_workers,
            use_gpu=False
        )
        return datamanager.train_loader, datamanager.num_train_pids

    elif mode == "eval":
        if domain_datasets in ["viper", "prid", "grid", "ilids"]:
            gallery_cls_size = 0
            RANDOM_SPLIT_TIMES = 10

            dummy_datamanager = ImageDataManager(
                root=data_dir,
                sources=domain_datasets,
                batch_size_test=1,
                workers=num_workers,
                use_gpu=False
            )

            gallery_cls_set = set()
            for _, pid, _, _ in dummy_datamanager.test_loader[domain_datasets]["gallery"]:
                gallery_cls_set.add(pid.item())
            gallery_cls_size = len(gallery_cls_set)
            del dummy_datamanager

            def _data_loader_iterator():
                for i in range(RANDOM_SPLIT_TIMES):
                    datamanager = ImageDataManager(
                        root=data_dir,
                        sources=domain_datasets,
                        split_id=i,
                        transforms=None,
                        batch_size_test=batch_size,
                        workers=num_workers,
                        use_gpu=False
                    )
                    yield (datamanager.test_loader[domain_datasets]["query"], datamanager.test_loader[domain_datasets]["gallery"])
            
            # FIXME
            return _data_loader_iterator, gallery_cls_size
            # return _data_loader_iterator, len(dummy_datamanager.test_loader[domain_datasets]["gallery"])



if __name__ == "__main__":
    from torchreid.data.datasets.image.viper import VIPeR

    datamanager = ImageDataManager(
        root="./data/datasets/",
        sources=["cuhk02", "dukemtmcreid", "market1501", "cuhk03"],
        # sources="prid",
        # targets=["viper"],
        split_id=9,
        # combineall=True,
        # train_sampler="RandomIdentitySampler",
        # num_instances=2,
        height=224,
        width=224,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        batch_size_test=10000,
        # workers=4
    )
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    for item in train_loader:
        pass
    
