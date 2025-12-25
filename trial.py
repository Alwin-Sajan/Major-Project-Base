import torch

# path = r"D:\MajorProject\basic-ui\models\best_resnet18_species.pth"

# # Load checkpoint
# ckpt = torch.load(path, map_location="cpu")
# print("Loaded checkpoint type:", type(ckpt))

# # Function to search for class labels in nested dicts
# def find_class_labels(d, prefix=""):
#     for k, v in d.items():
#         name = prefix + k

#         # Keys that commonly store classes
#         if any(tag in k.lower() for tag in ["class", "label", "category", "names"]):
#             print(f"\nPossible class info at: {name}")
#             print(v)

#         # Search inside nested dictionaries
#         if isinstance(v, dict):
#             find_class_labels(v, prefix=name + ".")

# If checkpoint is a dict (most model files)
# if isinstance(ckpt, dict):
#     print("\nSearching checkpoint for class labels...")
#     find_class_labels(ckpt)
# else:
#     print("Checkpoint is not a dict. Cannot inspect keys.")


from backend.train_resnet_embeddings import ResNet18Embedding, ClassifierHead, build_prototypes_from_loader, make_dataloaders


embedding_dim = 128
batch_size = 32
lr = 5e-4
epochs = 50
min_delta = 1e-4  # minimum meaningful improvement
data_root = "/media/abk/New Disk/DATASETS/first/updatedDataset" 

emb_net = ResNet18Embedding(embedding_dim=embedding_dim, use_cbam=True, pretrained=True)
clf = ClassifierHead(embedding_dim, 23)
train_loader, val_loader, classes = make_dataloaders(data_root, img_size=224, batch_size=batch_size)

ckpt = torch.load('models/best_supervised.pth', map_location="cuda")
emb_net.load_state_dict(ckpt['emb_state'])
clf.load_state_dict(ckpt['clf_state'])
emb_net.to("cuda"); emb_net.eval()

# Build prototypes from train set (or exemplar set)
prototypes = build_prototypes_from_loader(emb_net, train_loader, 23, device="cuda")

cosine_scores = torch.matmul(emb_net, prototypes.T)
print("Cosine scores:", cosine_scores.cpu().numpy())
