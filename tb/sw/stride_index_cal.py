import math

Z=765
W_s=5
P_r=51
P_c=1

U = math.ceil(Z / (P_r*W_s)) # number of stride units
B_s = math.ceil(Z / P_r) # B_{strd}: width of the stride boundary

# R^{u}_{i} \in \[0, N_{fg}\)^{P_{r}}
R = [ [[0]*P_r for i in range(W_s)] for u in range(U) ]
for u in range(U):
	for i in range(W_s):
		for j in range(P_r):
			R[u][i][j] = B_s*j+i+u*W_s
			print(f"R[{u}][{i}][{j}] = {R[u][i][j]}")

Z=15
W_s=1
P_r=3
P_c=1

# \hat{S}(i, j)
N_fg = P_r # number of stride fingers in a stride set
N_strd = math.ceil(Z / P_r) # number of stride sets where N_{fg} fingers in each stride set
S_i_j = 7
S_i_plus1_j = 7
hat_S_cur = S_i_j % N_strd
hat_S_next = S_i_plus1_j % N_strd
if hat_S_next >= hat_S_cur:
	hat_S_i_j = hat_S_next-hat_S_cur
else:
	hat_S_i_j = N_strd+hat_S_next-hat_S_cur
print(f"hat_S_i_j = {hat_S_i_j}")

# 1) Page address (index of stride set)
# 2) Word address (finger index in a stride set)
page_addr = [0]*Z
word_addr = [0]*Z
print(f"hat_S_cur = {hat_S_cur}")
# Step 1: To obtain the column index of the submatrix element
#         that will be shifted to the 0th column of the submatrix
I_new_col_0 = Z-S_i_j # I^{new}_{col_{0}}
print(f"I_new_col_0 = {I_new_col_0}")
for t in range(Z):
	# Step 2: To calculate the new column index that the submatrix element
	#         currently positioned in the (0+t)th column index, gets cyclic shifted to
	I_new_col_t = (I_new_col_0+t) % Z # I^{new}_{col_{t}}

	# Page address
	page_addr[t] = I_new_col_t % N_strd

	# Word address
	word_addr[t] = math.floor(I_new_col_t / N_strd)

	print(f"page_addr[{t}] = {page_addr[t]}, word_addr[{t}] = {word_addr[t]}")