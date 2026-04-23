import h5py, numpy as np, matplotlib.pyplot as plt

classes = [5, 6, 7, 8, 9]
NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar Round Smooth", "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]

#Read file
with h5py.File("Galaxy10_DECals.h5", "r") as f:
    images, labels = f["images"][:], f["ans"][:]

#Create subset
with h5py.File("galaxy_sample15.h5", "w") as out:
    idx = np.concatenate([np.random.choice(np.where(labels == c)[0], 3, replace=False) for c in classes])
    out.create_dataset("images", data=images[idx])
    out.create_dataset("ans", data=labels[idx])

#Read subset
with h5py.File("galaxy_sample15.h5", "r") as f:
    images, labels = f["images"][:], f["ans"][:]

#Visualize subset
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(NAMES[int(labels[i])])
    ax.axis("off")
plt.tight_layout(); plt.savefig("galaxy_grid.png", dpi=150); plt.show()
