import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

# Load the data
print("Loading DD_mat...")
DD_mat = torch.load("../data/DD_mat_IC_L.pt")
print("Loading val_pred.pkl...")
with open(
    "../experiments/FNN-DD_IC-3000-B128-lr0.0001-IC-4/val_pred.pkl", "rb"
) as f:
    data = pickle.load(f)

L_IC = data["true"][0].numpy()
pred_ep = data["pred"][0].numpy()
inputs = data["inputs"][0].numpy()

# Put the entries back to the matrix
DD = DD_mat["A"][0].numpy()
IC = DD_mat["L_IC"][0].numpy()

mask = np.abs(DD) != 0
mask_ic = np.abs(IC) != 0

DD_input = np.zeros_like(DD)[None, :, :]
DD_input = np.repeat(DD_input, pred_ep.shape[0], axis=0)

ic_placeholder = np.zeros_like(DD[None, :, :], dtype=L_IC.dtype).repeat(
    pred_ep.shape[0], axis=0
)
ep_placeholder = np.zeros_like(DD[None, :, :], dtype=L_IC.dtype).repeat(
    pred_ep.shape[0], axis=0
)


DD_input[:, mask] = inputs
ic_placeholder[:, mask_ic] = L_IC
ep_placeholder[:, mask_ic] = pred_ep

L_IC = ic_placeholder
pred_ep = ep_placeholder
L_pred = pred_ep

print(L_IC.shape, pred_ep.shape)

# Apply the IC
M_pred = np.linalg.inv(np.matmul(L_pred, L_pred.conj().transpose(0, 2, 1)))
M_IC = np.linalg.inv(np.matmul(L_IC, L_IC.conj().transpose(0, 2, 1)))
LLDD_pred = np.matmul(M_pred, DD_input)
LLDD_IC = np.matmul(M_IC, DD_input)

org_cond = []
L_pc_cond = []
IC_pc_cond = []

for i in range(DD_input.shape[0]):
    org_cond.append(np.linalg.cond(DD_input[i]))
    L_pc_cond.append(np.linalg.cond(LLDD_pred[i]))
    IC_pc_cond.append(np.linalg.cond(LLDD_IC[i]))

org_cond = np.array(org_cond)
L_pc_cond = np.array(L_pc_cond)
IC_pc_cond = np.array(IC_pc_cond)
sorted_idx = np.argsort(org_cond)

org_cond = org_cond[sorted_idx]
L_pc_cond = L_pc_cond[sorted_idx]
IC_pc_cond = IC_pc_cond[sorted_idx]
x = np.arange(org_cond.shape[0])

fig, ax = plt.subplots()
ax.set_box_aspect(1 / 1.62)
ax.scatter(x, org_cond, label="Original")
ax.scatter(x, L_pc_cond, label="NN precond")
ax.scatter(x, IC_pc_cond, label="IC precond")
ax.set_xlabel("Sample ID")
ax.set_ylabel("Condition number")
ax.set_ylim(0, 1000)
# ax.set_yscale('log')
ax.legend()
ax.grid(linestyle="dotted")
fig.savefig(
    "../../docs/updates/figs/supervised-L-condition-number-compare.png",
    format="png",
    dpi=500,
    bbox_inches="tight",
)
plt.show()
