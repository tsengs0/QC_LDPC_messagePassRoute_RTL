import math
import numpy as np

def page_addr_gen(t: int, Z: int, shift_factor: int, N_rc: int):
	I_new_col_0 = Z-shift_factor
	I_new_col_t = (I_new_col_0+t) % Z
	page_addr=I_new_col_t % N_rc
	word_addr=math.floor(I_new_col_t / N_rc)
	return page_addr, word_addr

#--------------------
#Z=765
#W_s=5
#P_r=51
#P_c=1
#--------------------
Z=15
W_s=1
P_r=3
P_c=1

U = math.ceil(Z / (P_r*W_s)) # number of stride units
N_rc = math.ceil(Z / P_r) # number of row chunks in a submatrix, each of which contains N_{fg} stride fingers
msgPass_buffer = np.zeros(shape=(N_rc*2+1, P_r), dtype=np.int32) # Depth: Region 0) N_rc num. of compV2C row chunks
																 #        Region 1) a blank row chunk as separator
																 #        Region 2) N_rc num. of permV2C row chunks
print(f"size of msgPass_buffer = {msgPass_buffer.shape}")
# R^{u}_{i} \in \[0, N_{fg}\)^{P_{r}}
R = [ [[0]*P_r for i in range(W_s)] for u in range(U) ]
for u in range(U):
	for i in range(W_s):
		for j in range(P_r):
			R[u][i][j] = N_rc*j+i+u*W_s
			print(f"R[{u}][{i}][{j}] = {R[u][i][j]}")

# \hat{S}(i, j)
N_fg = P_r # number of stride fingers in a stride set
N_rc = math.ceil(Z / P_r) # number of row chunks in a submatrix, each of which contains N_{fg} stride fingers
S_i_j = 0
S_i_plus1_j = 7
hat_S_cur = S_i_j % N_fg
hat_S_next = S_i_plus1_j % N_fg
if hat_S_next >= hat_S_cur:
	hat_S_i_j = hat_S_next-hat_S_cur
else:
	hat_S_i_j = N_fg+hat_S_next-hat_S_cur
print(f"hat_S_i_j = {hat_S_i_j}")

# 1) Page address (index of stride set)
# 2) Word address (finger index in a stride set)
page_addr = [0]*Z
word_addr = [0]*Z
print(f"hat_S_cur = {hat_S_cur}")
# Step 1: To obtain the column index of the submatrix element
#         that will be shifted to the 0th column of the submatrix
#I_new_col_0 = Z-S_i_j # I^{new}_{col_{0}}
#print(f"I_new_col_0 = {I_new_col_0}")
for t in range(Z):
	# Step 2: To calculate the new column index that the submatrix element
	#         currently positioned in the (0+t)th column index, gets cyclic shifted to
	#I_new_col_t = (I_new_col_0+t) % Z # I^{new}_{col_{t}}

	# Page address
	#page_addr[t] = page_addr_gen(I_new_col_t=I_new_col_t, N_rc=N_rc)#I_new_col_t % N_rc

	# Word address
	#word_addr[t] = word_addr_gen(I_new_col_t=I_new_col_t, N_rc=N_rc)#math.floor(I_new_col_t / N_rc)

	page_addr[t], word_addr[t] = page_addr_gen(t=t, Z=Z, shift_factor=S_i_j, N_rc=N_rc)

	msgPass_buffer[ page_addr[t] ][ word_addr[t] ] = t
	print(f"page_addr[{t}] = {page_addr[t]}, word_addr[{t}] = {word_addr[t]}, msgPass_buffer[ page_addr[t] ][ word_addr[t] ] = {msgPass_buffer[ page_addr[t] ][ word_addr[t] ]}")
#------------------------------------------------------------------------
compV2C_base_addr = 0 # tentative value for ease of simulation
permV2C_base_addr = N_rc+1 # tentative value for ease of simulation
page_raddr = 0 + compV2C_base_addr

S_bs = hat_S_i_j # Cyclic shift factor as the shift control signal for the L1BS
print(f"msgPass_buffer before cyclic shift = \n{msgPass_buffer}")
print("------------------------------------------------------------------------")
for u in range(U):
	for i in range(W_s):
		t = R[u][i][0] = N_rc*j+i+u*W_s
		page_waddr, word_waddr = page_addr_gen(t=t, Z=Z, shift_factor=S_i_plus1_j, N_rc=N_rc)
		hat_i = page_waddr
		hat_i += permV2C_base_addr
		for j in range(P_r):
			y_fetch_compV2C = msgPass_buffer[page_raddr][j]
			hat_j = int((j + S_bs) % P_r)
			print(f"S^bs={S_bs}, page_raddr={page_raddr}, j={j} --> hat_i = {hat_i}, hat_j = {hat_j}, y_fetch_compV2C = {y_fetch_compV2C}")
			msgPass_buffer[hat_i][hat_j] = y_fetch_compV2C
		page_raddr += 1

print(f"msgPass_buffer after cyclic shift = \n{msgPass_buffer}")
