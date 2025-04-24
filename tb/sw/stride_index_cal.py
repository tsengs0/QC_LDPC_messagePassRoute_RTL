#!/usr/bin/env python
# coding: utf-8

# To import necessary Python packages.

# In[1017]:


import math
import numpy as np
import pandas as pd
from typing import List
import gc
from collections import defaultdict


# Global configuration for the subsequent simulation/emulation environment

# In[1018]:


global DEBUG_VERBOSE_LEVEL_0
global DEBUG_VERBOSE_LEVEL_1
DEBUG_VERBOSE_LEVEL_0 = False
DEBUG_VERBOSE_LEVEL_1 = False


# ## Page and word addresses generation

# Let $S(i, j)$ and $S(i+1, j)$ denote the shfit factors of a given set of submatrices.
# The Z messages associated with a submatrix are mapped to a bunch of row chunks in a stride-pattern fashion. For each message, $t \in \{0, 1, \cdots, Z-1\}$, its corresponding row chunk index is calculated by
# $$page\_addr(t, i, j)=I^{new}_{col_{t}} \pmod {N_{rc}}$$
# 
# where
# $$I^{new}_{col_{t}} = (I^{new}_{col_{0}}+t) \pmod{Z}$$
# $$I^{new}_{col_{0}} = Z - S(i, j)$$
# 
# Furthermore, each aforementioned message $t$ is located in a certain orow chunk by a word address, that is
# $$word\_addr(t, i, j) =\lfloor I^{new}_{col_t} / N_{rc} \rfloor$$

# In[1019]:


layerSh_rom_page_dtype_1 = np.dtype([
	('stride_unit', np.int32),
	('rel_row_chunk', np.int32)
])
class msgPass_buffer_ctrl:
	I_new_col_0 = None
	I_new_col_t = None

	def __init__(self, msgPass_sched: str, baseMatrix_row_num: int, baseMatrix_col_num: int, Z: int, N_rc: int, U: int, W_s: int, P_c: int, P_r: int):
		self.msgPass_sched = msgPass_sched
		self.baseMatrix_row_num = baseMatrix_row_num
		self.baseMatrix_col_num = baseMatrix_col_num
		self.Z = Z
		self.N_rc = N_rc
		self.U = U
		self.W_s = W_s
		self.P_c = P_c
		self.P_r = P_r

		self.memBlk_page_depth = math.ceil(self.baseMatrix_col_num / self.P_c)*self.U
		self.shift_factor_rom = np.zeros(
			shape=(
				self.P_c, # number of parallel base-matrix columns
				self.W_s, # number of MEM blocks in a parallel base-matrix column
				self.baseMatrix_row_num, # number of memory regions allocated for respective decoding layers
	    		self.memBlk_page_depth # number of write-back patterns for each MEM block, i.e. memory depth
			),
			dtype=np.int32
		)

		self.page_waddr_vec = np.zeros(
			shape=(
				self.P_c, # number of parallel base-matrix columns
				self.W_s, # number of MEM blocks in a parallel base-matrix column
				self.baseMatrix_row_num, # number of memory regions allocated for respective decoding layers
	    		self.memBlk_page_depth # number of write-back patterns for each MEM block, i.e. memory depth
			),
			dtype=layerSh_rom_page_dtype_1
		)

	def new_col_normalisation(self, t: int, shift_factor: int):
		self.I_new_col_0 = self.Z-shift_factor
		self.I_new_col_t = (self.I_new_col_0+t) % self.Z

	def page_addr_gen(self, t: int, shift_factor: int):
		self.new_col_normalisation(t=t, shift_factor=shift_factor)
		if(self.msgPass_sched=="stride_sched"):
			page_addr=self.I_new_col_t % self.N_rc
		return page_addr

	def word_addr_gen(self, t: int, shift_factor: int):
		self.new_col_normalisation(t=t, shift_factor=shift_factor)
		if(self.msgPass_sched=="stride_sched"):
			word_addr=math.floor(self.I_new_col_t / self.N_rc)
		return word_addr	



# ## Configuration of the decoder architecture
# - Parallelism in rows and columns of the target parity-check matrix (expanded from a given base matrix)
# - To instantiate a container to emulate the message-pass buffer

# In[1020]:


deg_chk_max=10
deg_var_max=3
baseMatrix_row_num = deg_var_max
baseMatrix_col_num = deg_chk_max
Z=765
W_s=5
P_r=51
P_c=2
S_i_j=np.array([
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	[0, 656, 76, 132, 184, 233, 216, 490, 714, 715],
	[0, 22, 650, 359, 587, 65, 463, 635, 91, 100]],

    dtype=np.int32
)

U = math.ceil(Z / (P_r*W_s)) # number of stride units
N_rc = math.ceil(Z / P_r) # number of absolute row chunks in a submatrix, each of which contains N_{fg} stride fingers
msgPass_buffer_norm = np.zeros(shape=(N_rc*2+1, P_r), dtype=np.int32) # Depth: Region 0) N_rc num. of compV2C row chunks
																      #        Region 1) a blank row chunk as separator
																      #        Region 2) N_rc num. of permV2C row chunks

# A virtual message-pass buffer controller to generate the page and word addresses for writing back the cyclic shifted messages
msgPass_buffer_ctrl_inst = msgPass_buffer_ctrl(
    msgPass_sched="stride_sched",
    baseMatrix_row_num=baseMatrix_row_num,
    baseMatrix_col_num=baseMatrix_col_num,
    Z=Z,
    N_rc=N_rc,
    U=U,
    W_s=W_s,
    P_r=P_r,
    P_c=P_c
)


# 

# In[1021]:


# 1) Page address (index of stride set)
# 2) Word address (finger index in a stride set)
page_addr = [0]*Z
word_addr = [0]*Z

def msgPass_buffer_permMsg_write(
		msgPass_sched: str,
		compMsg_vec: List[int], # Set of computed messages before getting (cyclic) shifted
		Z: int,
		shift_factor: int,
		N_rc: int,
		msgPass_buffer_inst: List[ List[int] ],
		permMsg_pageAddr_base: int # Base address of permuted messages region in msgPass_buffer
) -> List[ List[int] ]:
	for t in compMsg_vec:
		page_addr[t] = msgPass_buffer_ctrl_inst.page_addr_gen(t=t, shift_factor=shift_factor)
		word_addr[t] = msgPass_buffer_ctrl_inst.word_addr_gen(t=t, shift_factor=shift_factor)
		msgPass_buffer_inst[ page_addr[t]+permMsg_pageAddr_base ][ word_addr[t] ] = t
	return msgPass_buffer_inst


# Let test an example that $S(i,j)=2$ and $S(i+1,j)=7$.

# In[1022]:


initial_compMsg_vec = [t for t in range(Z)]

permV2C_base_addr_vec = [0]*2
permV2C_base_addr_vec[0] = 0 # tentative value for ease of simulation
permV2C_base_addr_vec[1] = N_rc+1 # tentative value for ease of simulation

# For S(i, j)
msgPass_buffer_norm = msgPass_buffer_permMsg_write(
		msgPass_sched="stride_sched",
		compMsg_vec=initial_compMsg_vec,
		Z=Z,
		shift_factor=S_i_j[0][1],
		N_rc=N_rc,
		msgPass_buffer_inst=msgPass_buffer_norm,
		permMsg_pageAddr_base=permV2C_base_addr_vec[0]
)

# For S(i+1, j)
msgPass_buffer_norm = msgPass_buffer_permMsg_write(
		msgPass_sched="stride_sched",
		compMsg_vec=initial_compMsg_vec,
		Z=Z,
		shift_factor=S_i_j[1][1],
		N_rc=N_rc,
		msgPass_buffer_inst=msgPass_buffer_norm,
		permMsg_pageAddr_base=permV2C_base_addr_vec[1]
)

#print(msgPass_buffer_norm)


# The next section will address the cyclic shfit factor for each row chunk.

# # Stride Unit Assignment
# 
# The row chunks, stride units and input sources of a page alignment unit are formulated as follows. A submatrix $b$ consists of a set of stride units, $\mathbb{S}=\{\mathbb{S}_{0}, \mathbb{S}_{1}, \cdots. \mathbb{S}_{U-1}\}$ where $U = Z / (P_{r} \cdot W^{s})$ accounts for the number of stride units. For all $u \in \{0, \cdots, U-1\}$, a set of consecutive row chunks is included in a stride unit, i.e. $\mathbb{S}_{u} = \{R^{u}_{0}, R^{u}_{2}, \cdots, R^{u}_{W^{s}-1}\}$ where $R^{u}_{0}$ denotes the first row chunk in the $u$th stride unit, etc. Moreover, a row chunk $R^{u}_{i}$ aggregates a bunch of extrinsic messages\footnote{Every extrinsic message is from a nonzero element in submatrix $b$, which represents one associated variable node.}, i.e.
# 
# $$
# \forall i \in \{0, \cdots, W^{s}-1\}, \forall u \in \{0, \cdots, U-1\}, \\
# R^{u}_{i} = \{y_{o} | 0 \leq j \leq P_{r}-1, j \in \mathbb{Z}^{+}, o=N_{rc}*j+i+u*W^{s}\}.
# $$

# # Generation of Row-Chunk Cyclic Shift Factors
# 
# This section is to determine the cyclic shift factor for the permutation of each row chunk.
# Let pick up one element from each row chunk as representatives, presenting by a set $E^{pg}$,
# $$
# E^{pg} = \{t \in [0, Z) | t \pmod {P_{r}}=0\}.
# $$
# 
# Next step is to get the page address and word address from each element of $E^{pg}$,
# $$
# \forall e \in E^{pg}, \\
# \hat{P}^{cur}_{e} = page\_addr(e, i, j), \\
# \hat{P}^{next}_{e} = page\_addr(e, i+1, j), \\
# \hat{W}^{cur}_{e} = word\_addr(e, i, j), \\
# \hat{W}^{next}_{e} = word\_addr(e, i+1, j). \\
# $$
# 
# Finally, the cyclic shift factor for passing messages in each row chunk, from $i$-th decoding layer to $(i+1)$-th deocoding layer, can be calculated by
# $$
# \hat{S}(i, j, \hat{P}^{cur}_{e}) = 
# \begin{cases}
#     \hat{W}^{next}_{e} - \hat{W}^{cur}_{e}, & \text{if } \hat{W}^{next}_{e} \ge \hat{W}^{cur}_{e} \\
#     N_{rc} + \hat{W}^{next}_{e} - \hat{W}^{cur}_{e}, & \text{otherwise}
# \end{cases}
# $$

# Example of $submatrix(i=0, j=0)$ and $submatrix(i=1, j=0)$.

# In[1023]:


abs_rowChunk_shfit_factor_vec = np.zeros(
    shape=(
        baseMatrix_row_num,
        baseMatrix_col_num,
        N_rc # number of row chunks in a submatrix
    ),
    dtype=np.int32
)

page_vec = [t for t in range(N_rc)]

for bm_row in range(baseMatrix_row_num):
    for bm_col in range(baseMatrix_col_num):
        for t in page_vec:
            # To generate the cyclic shift factors for row chunks
            p0 = msgPass_buffer_ctrl_inst.page_addr_gen(t=t, shift_factor=S_i_j[bm_row][bm_col])
            w0 = msgPass_buffer_ctrl_inst.word_addr_gen(t=t, shift_factor=S_i_j[bm_row][bm_col])
            p1 = msgPass_buffer_ctrl_inst.page_addr_gen(t=t, shift_factor=S_i_j[(bm_row+1) % baseMatrix_row_num][bm_col])
            w1 = msgPass_buffer_ctrl_inst.word_addr_gen(t=t, shift_factor=S_i_j[(bm_row+1) % baseMatrix_row_num][bm_col])
            if w1 < w0:
                hat_S_i_j_r = P_r+w1-w0
            else: # w1 >= w0
                hat_S_i_j_r = w1-w0
            #print(f"submatrix({bm_row}, {bm_col}) \t->\t hat_P_cur_e={p0}\t->\t{p0}-th row chunk gets shifted by the cyclic shift factor {hat_S_i_j_r}, p1: {p1}")
            abs_rowChunk_shfit_factor_vec[bm_row][bm_col][p0] = hat_S_i_j_r

#print(f"abs_rowChunk_shfit_factor_vec: {abs_rowChunk_shfit_factor_vec}")


# # Algorithm - Message Passing Procedures for the Computed V2C Messages

# - StrideUnit-to-message converter: to obtain the indices of stride unit and relative row chunk that $t$-th message is mapped to 
# $$
# \forall i \in \{0, \cdots, W^{s}-1\}, \forall u \in \{0, \cdots, U-1\}, \\
# R^{u}_{i} = \{y_{o} | 0 \leq j \leq P_{r}-1, j \in \mathbb{Z}^{+}, o=N_{rc}*j+i+u*W^{s}\}.
# $$
# 
# - Message-to-StrideUnit converter, assume the cyclic shift factor is zero
#   - Input: message index $t \in [0, Z)$
#   - Output 1): Stride unit index, $u$
#   - OUtput 2): Relative row chunk index, $i$
# $$
# \forall t \in [0, Z), \\
# 
# u = \lfloor (t \mod{(W^{s} \cdot U)}) / W^{s} \rfloor \\
# i = (t \mod{(W^{s} \cdot U)}) \mod W^{s}
# $$
# 
# ---
# 
# Let assign sample values to all stride unit.

# In[1024]:


R = [ [[0]*P_r for i in range(W_s)] for u in range(U) ]
for u in range(U):
	for i in range(W_s):
		for j in range(P_r):
			R[u][i][j] = N_rc*j+i+u*W_s
			#print(f"R[{u}][{i}][{j}] = {R[u][i][j]}")

for t in range(Z):
	u = math.floor((t % (W_s*U)) / W_s)
	i = (t % (W_s*U)) % W_s
	#print(f"t={t} u={u} i={i}")
	if t not in R[u][i]:
		print(f"Error: t={t} u={u} i={i} R[u][i]={R[u][i]}") # Error: t is not in the list of R[u][i]


# # Emulation of Layer-Shift ROM
# 
# The Layer-shift ROM contains two memory regions storing: 1) the cyclic shift factors, and 2) page addresses for the cyclic shifted row chunks
# 
# 1. To convert the vector of cyclic shift factors permuted by the indices of the absolute row chunks, into the vector of cyclic shift shift factors permuted by the indices of the parallel base-matrix columns, the stride units and the relative row chunks.
# 
# 2. To precalculate the page addresses for writing back the cyclic shifted message within $submatrix(i,j)$
#     - There are $W^{s}$ number of memory blocks with independent controls, where the messages from every stride unit of row chunks are evenly distributed over those memory blocks. There are $U=\lceil Z / (P_{r} \cdot W^{s}) \rceil$ stride units containing $W^{s}$ row chunks for each. Therefore, a total of $\lceil Z / P_{r} \rceil$ row chunks labelled by absolute indices in a submatrix need to have corresponding relative row-chunk indices and stride unit indices as follows.
# $$
# \forall r \in [0, \lceil Z / P_{r} \rceil), \\
# r^{rel}(r) = r^{abs}_{r} \mod W^{s} \\
# u(r) = \lfloor r^{abs}_{r} / W^{s} \rfloor
# $$

# In[1025]:


page_vec = [t for t in range(N_rc)]
for bm_row in range(baseMatrix_row_num):
    for bm_col in range(baseMatrix_col_num):
        # Initial Step:
        abs_rowChunk_id_vec     = list()
        rel_rowChunk_id_vec     = list()
        stride_unit_id_vec      = list()
        hat_abs_rowChunk_id_vec = list()
        hat_rel_rowChunk_id_vec = list()
        hat_stride_unit_id_vec  = list()
        for t in page_vec:
            abs_rowChunk_id = msgPass_buffer_ctrl_inst.page_addr_gen(t=t, shift_factor=S_i_j[bm_row][bm_col])
            rel_rowChunk_id = abs_rowChunk_id % W_s
            stride_unit_id = math.floor(abs_rowChunk_id / W_s)
            abs_rowChunk_id_vec.append(abs_rowChunk_id)
            rel_rowChunk_id_vec.append(rel_rowChunk_id)
            stride_unit_id_vec.append(stride_unit_id)
            #print(f"submatrix({bm_row}, {bm_col})_codebit: {t}|\t->\t| Row chunk for i-th layer:\t {abs_rowChunk_id} (absolute) at stride unit {stride_unit_id} converted to {rel_rowChunk_id} (relative) at stride unit {stride_unit_id}")

            hat_abs_rowChunk_id = msgPass_buffer_ctrl_inst.page_addr_gen(t=t, shift_factor=S_i_j[(bm_row+1) % baseMatrix_row_num][bm_col])
            hat_rel_rowChunk_id = hat_abs_rowChunk_id % W_s
            hat_stride_unit_id = math.floor(hat_abs_rowChunk_id / W_s)
            hat_abs_rowChunk_id_vec.append(hat_abs_rowChunk_id)
            hat_rel_rowChunk_id_vec.append(hat_rel_rowChunk_id)
            hat_stride_unit_id_vec.append(hat_stride_unit_id)
            #print(f"submatrix({(bm_row+1)%baseMatrix_row_num}, {bm_col})_codebit: {t}|\t->\t| Row chunk for (i+1)-th layer:\t {hat_abs_rowChunk_id} (absolute) at stride unit {hat_stride_unit_id} converted to {hat_rel_rowChunk_id} (relative) at stride unit {hat_stride_unit_id}")



        # 1) absolute-to-relative row-chunk shift factor conversion
        # abs_rowChunk_shfit_factor_vec = np.zeros(
        #     shape=(
        #         baseMatrix_row_num,
        #         baseMatrix_col_num,
        #         N_rc # number of row chunks in a submatrix
        #     ),
        #     dtype=np.int32
        # )
        #                  |
        #                  |
        #                  |
        #                  V
        # shift_factor_rom = np.zeros(
        # 	shape=(
        # 		self.P_c, # number of parallel base-matrix columns
        # 		self.W_s, # number of MEM blocks in a parallel base-matrix column
        # 		self.baseMatrix_row_num, # number of memory regions allocated for respective decoding layers
        # 		self.memBlk_page_depth, # number of write-back patterns for each MEM block, i.e. memory depth
        # 	),
        # 	dtype=layerSh_rom_page_dtype_0
        # )
        rel_bm_col = bm_col % P_c # baseMatrix_col_num number of base-matrix columns are evenly allocated to
                                  # the processes of those P_c number of paralle base-matrix columns.
                                  # Namely, for {baseMatrix_col_num=4, P_c=2}, the base-matrix columns' corresponding
                                  # message-passing operations are handled by the following parallel base-matrix column
                                  # deployment:
                                  #   a) Parallel (relative) base-matrix column 0: handles base-matrix columns {0, 2}
                                  #   b) Parallel (relative) base-matrix column 1: handles base-matrix columns {1, 3}
        page_addr_base = math.floor(bm_col / P_c)*U
        for abs_row_chunk in range(N_rc):
            rel_rowChunk_id = abs_row_chunk % W_s
            stride_unit = math.floor(abs_row_chunk / W_s)
            #print(f"msgPass_buffer_ctrl_inst.shift_factor_rom[{rel_bm_col}][{rel_rorel_rowChunk_idwChunk_id}][{bm_row}][{stride_unit}+{page_addr_base}]: {abs_rowChunk_shfit_factor_vec[bm_row][bm_col][abs_row_chunk]}")
            msgPass_buffer_ctrl_inst.shift_factor_rom[rel_bm_col][rel_rowChunk_id][bm_row][stride_unit+page_addr_base] = abs_rowChunk_shfit_factor_vec[bm_row][bm_col][abs_row_chunk]

        # 2) To generate the page addresses for writing back the cyclic shifted messages
        rel_bm_col = bm_col % P_c # baseMatrix_col_num number of base-matrix columns are evenly allocated to
                                  # the processes of those P_c number of paralle base-matrix columns.
                                  # Namely, for {baseMatrix_col_num=4, P_c=2}, the base-matrix columns' corresponding
                                  # message-passing operations are handled by the following parallel base-matrix column
                                  # deployment:
                                  #   a) Parallel (relative) base-matrix column 0: handles base-matrix columns {0, 2}
                                  #   b) Parallel (relative) base-matrix column 1: handles base-matrix columns {1, 3}
        page_addr_base = math.floor(bm_col / P_c)*U
        for abs_row_chunk in range(N_rc):
            cur_stride_unit = stride_unit_id_vec[abs_row_chunk]
            hat_stride_unit = hat_stride_unit_id_vec[abs_row_chunk]
            rel_rowChunk_id = rel_rowChunk_id_vec[abs_row_chunk]
            msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_rowChunk_id][bm_row][cur_stride_unit+page_addr_base]['stride_unit'] = hat_stride_unit
            msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_rowChunk_id][bm_row][cur_stride_unit+page_addr_base]['rel_row_chunk'] = hat_rel_rowChunk_id_vec[abs_row_chunk]
            #print(f"Submatrix({bm_row}, {bm_col}), StrideUnit_{cur_stride_unit}, RelRowChunk_{rel_rowChunk_id} -> StrideUnit_{hat_stride_unit}, RelRowChunk_{hat_rel_rowChunk_id_vec[abs_row_chunk]}: {msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_rowChunk_id][bm_row][cur_stride_unit+page_addr_base]}")

        del abs_rowChunk_id_vec
        del rel_rowChunk_id_vec
        del stride_unit_id_vec
        del hat_abs_rowChunk_id_vec
        del hat_rel_rowChunk_id_vec
        del hat_stride_unit_id_vec
        gc.collect() # For garbage collection



# Let's confirm the content of the generated cyclic shift factors and page waddr values.

# In[1026]:


for bm_row in range(baseMatrix_row_num):
    #print(f"-----------\nLayer {bm_row}")
    for bm_col in range(baseMatrix_col_num):
        #print(f"\t#Base-matrix column {bm_col}")
        rel_bm_col = bm_col % P_c
        page_addr_base = math.floor(bm_col / P_c)*U
        for stride_unit in range(U):
            #print(f"\t\t#Stride unit {stride_unit}")
            for rel_row_chunk in range(W_s):
                stride_unit_waddr = msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_row_chunk][bm_row][stride_unit+page_addr_base]['stride_unit']
                rc_waddr = msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_row_chunk][bm_row][stride_unit+page_addr_base]['rel_row_chunk']
                cyclic_shift_factor = msgPass_buffer_ctrl_inst.shift_factor_rom[rel_bm_col][rel_row_chunk][bm_row][stride_unit+page_addr_base]
                #print(f"\t\t\tRelative row chunk {rel_row_chunk} "+
                #      f"page_waddr (strideUnit, rel_row_chunk): ({stride_unit_waddr}, {rc_waddr}),\tcyclic shift factor: {cyclic_shift_factor}")


# # Simulation of the message passing operation of the computed variable-to-check messages

# ---
# 
# Declaration of the message-pass buffer composed of $W^{s}$ number of memory blocks for each parallel base-matrix column.

# In[1027]:


memBlk_page_depth = math.ceil(baseMatrix_col_num / P_c)*U*(baseMatrix_row_num+1) # "+1" at the last operand, "(baseMatrix_row_num+1)",
																				 # is to preserve one more memory region to store the 
                                                                                 # initial permutation for the dump verification
msgPass_buffer = np.zeros(
	shape=(
	    P_c, # number of parallel base-matrix columns
	    W_s, # number of MEM blocks in a parallel base-matrix column
        baseMatrix_row_num+1, # number of memory regions allocated for respective decoding layers, with an additional layer preserved for verification
	    memBlk_page_depth, # number of write-back patterns for each MEM block, i.e. memory depth
        P_r # number of messages packed in one memory page
	),
	dtype=np.int32
)


# To preload the sample channel messages (input LLR values) to the message-pass buffers where the message positions are all aligned with the cyclic shift consequences of the submatrices in the first decoding layer.
# 
# The input LLR loader is implemented by the following steps:
# 1. To generate the $P_{c}$ sets of input LLR messages, each of which contains $Z$ messages assigned with the values: $\{0, 1, \cdots, Z-1\}$.
# 2. To rearrange the those two sets' permutations according to the cyclic shift factors of their respective submatrices.
# 3. To load the cyclic shifted LLR messages to the corresponding memory blocks.

# In[1028]:


for bm_row in range(baseMatrix_row_num+1):
    for bm_col in range(baseMatrix_col_num):
        #in_llr_vec = np.arange(0, Z, dtype=np.int32)
        #in_llr_vec = np.roll(in_llr_vec, S_i_j[bm_row][bm_col])
        #print(f"submatrix({bm_row}, {bm_col}) in_llr_vec={in_llr_vec}")

        rel_bm_col = bm_col % P_c
        rel_bm_col_cnt = math.floor(bm_col / P_c)
        page_addr_base = rel_bm_col_cnt*U
        for t in range(Z):
            u = math.floor((t % (W_s*U)) / W_s)
            rel_row_chunk = (t % (W_s*U)) % W_s
            word_addr = msgPass_buffer_ctrl_inst.word_addr_gen(t=t, shift_factor=0)
            msgPass_buffer[rel_bm_col][rel_row_chunk][bm_row][u+page_addr_base][word_addr] = t
            #print(f"msgPass_buffer[rel_bm_col:{rel_bm_col}][rel_row_chunk:{rel_row_chunk}][bm_row:{bm_row}][page_addr:strideUnit_{u}+base_addr_{page_addr_base}][word_addr:{word_addr}]={t}")


# # Ongoing: the MUX4 should be inserted before this line:
# ```
# y_fetch_compV2C = msgPass_buffer[rel_bm_col][rel_row_chunk][layer_next][stride_unit+rel_bm_col_cnt][j]
# ```

# In[1029]:


#compV2C_base_addr = 0 # tentative value for ease of simulation
#permV2C_base_addr = N_rc+1 # tentative value for ease of simulation
#page_raddr = 0 + compV2C_base_addr
prefix_for_verification = 10**(math.ceil(math.log10(Z-1)))

def msgPass_stride_sched(
	layer: int,
	layer_next: int,
	rel_bm_col: int, # Relative base-matrix column in {0, ..., P_{c}-1}
	rel_bm_col_cnt: int # To count the number of base-matrix columns which have been processed by the underlying message-pass routing network
						# Note that P_{c} number of independent message-pass routing networks deployed on the decoder subsystem
):
	page_raddr = 0
	page_addr_base = rel_bm_col_cnt*U # Each base-matrix column requires U pages of row chunk messages on every memory block
	for stride_unit in range(U):
		for rel_row_chunk in range(W_s):
			S_bs = msgPass_buffer_ctrl_inst.shift_factor_rom[rel_bm_col][rel_row_chunk][layer][stride_unit+page_addr_base]
			#print(f"\t\t\trel_bm_col_cnt: {rel_bm_col_cnt}, strideUnit_{stride_unit}, rel_row_chunk_{rel_row_chunk}, S_bs: {S_bs}")
			stride_unit_waddr = msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_row_chunk][layer][stride_unit+page_addr_base]['stride_unit']
			rc_waddr = msgPass_buffer_ctrl_inst.page_waddr_vec[rel_bm_col][rel_row_chunk][layer][stride_unit+page_addr_base]['rel_row_chunk']
			for j in range(P_r):
				# Fetch the compV2C
				y_fetch_compV2C = msgPass_buffer[rel_bm_col][rel_row_chunk][layer][stride_unit+rel_bm_col_cnt*U][j]

				# Column-wise cyclic shift outputing the permV2C
				hat_j = int((j + S_bs) % P_r)

				# Wirte back the permV2C
				msgPass_buffer[rel_bm_col][rc_waddr][layer_next][stride_unit_waddr+rel_bm_col_cnt*U][hat_j] = y_fetch_compV2C+prefix_for_verification*layer

			# == For fine-grained debug (1) =========================================================================================
			if(DEBUG_VERBOSE_LEVEL_1==True):
				print(f"\t\t\t\tFetch compV2Cs from msgPass_buffer[{rel_bm_col}][{rel_row_chunk}][{layer}][{stride_unit+rel_bm_col_cnt*U}]")
				temp_pre = msgPass_buffer[rel_bm_col][rel_row_chunk][layer][stride_unit+rel_bm_col_cnt*U]
				temp_post = msgPass_buffer[rel_bm_col][rc_waddr][layer_next][stride_unit_waddr+rel_bm_col_cnt*U]
				print(f"\t\t\t\tS_bs: {S_bs}, compV2C_vec: {temp_pre} -> write permV2C_vec: {temp_post} back to memblk[{rc_waddr}], layer-{layer_next}, rel_bm_col-{rel_bm_col}, strideUnit-{stride_unit_waddr} w/ base_addr-{rel_bm_col_cnt*U}")
			# == End of fine-grained debug (1) ======================================================================================
			page_raddr += 1

def msgPass_buffer_dump(layer:int, rel_bm_col: int, rel_bm_col_cnt: int):
	print(f"\tLayer-{layer}, Relative Base-Matrix Column_{rel_bm_col} w/ base region_{rel_bm_col_cnt}")
	page_addr_base = rel_bm_col_cnt*U
	for stride_unit in range(U):
		print(f"\t\tStride unit {stride_unit}")
		for memblk in range(W_s):
			print(f"\t\t\tmemblk_{memblk} -> {msgPass_buffer[rel_bm_col][memblk][layer][stride_unit+page_addr_base]}")

def msgPass_buffer_reset_with_x(layer:int, rel_bm_col: int, rel_bm_col_cnt: int):
	# Let define value Z as the X value (considered as uncertainty or noise)
	page_addr_base = rel_bm_col_cnt*U
	for stride_unit in range(U):
		for memblk in range(W_s):
			msgPass_buffer[rel_bm_col][memblk][layer][stride_unit+page_addr_base]=Z

if(DEBUG_VERBOSE_LEVEL_0==True):
	print("\n------------------------------------------------------------------------")
	print(f"msgPass_buffer before cyclic shift")
	for layer in range(baseMatrix_row_num):
		for rel_bm_col_cnt in range(math.ceil(baseMatrix_col_num / P_c)):
			for rel_bm_col in range(P_c):
				msgPass_buffer_dump(layer=layer, rel_bm_col=rel_bm_col, rel_bm_col_cnt=rel_bm_col_cnt)
	print("------------------------------------------------------------------------")

for layer in range(baseMatrix_row_num):
	layer_next = (layer+1) % baseMatrix_row_num
	for bm_col in range(baseMatrix_col_num):
		rel_bm_col = bm_col % P_c
		rel_bm_col_cnt = math.floor(bm_col / P_c)
		msgPass_stride_sched(layer=layer, layer_next=layer_next, rel_bm_col=rel_bm_col, rel_bm_col_cnt=rel_bm_col_cnt)

if(DEBUG_VERBOSE_LEVEL_0==True):
	print("\n------------------------------------------------------------------------")
	print(f"msgPass_buffer after cyclic shift:")
	for layer in range(baseMatrix_row_num):
		layer_next = (layer+1) % baseMatrix_row_num
		for rel_bm_col_cnt in range(math.ceil(baseMatrix_col_num / P_c)):
			for rel_bm_col in range(P_c):
				msgPass_buffer_dump(layer=layer_next, rel_bm_col=rel_bm_col, rel_bm_col_cnt=rel_bm_col_cnt)


# # Verification of the Simulation Results

# In[1030]:


def msgPass_buffer_dump_verification(layer_num: int, first_layer: int, last_layer: int, rel_bm_col: int, rel_bm_col_cnt: int) -> bool:
	is_verification_passed = True
	prefix_for_lastLayer_verification = prefix_for_verification*layer_num
	page_addr_base = rel_bm_col_cnt*U
	for stride_unit in range(U):
		for memblk in range(W_s):
			if((msgPass_buffer[rel_bm_col][memblk][first_layer][stride_unit+page_addr_base]<prefix_for_lastLayer_verification).any()):
				print(f"(Loss of message passing) Layer-{layer}, Relative Base-Matrix Column_{rel_bm_col} w/ base region_{rel_bm_col_cnt}, Stride unit {stride_unit}, memblk_{memblk} -> {msgPass_buffer[rel_bm_col][memblk][first_layer][stride_unit+page_addr_base]}")
				is_verification_passed=False

			if(
				np.not_equal(
					msgPass_buffer[rel_bm_col][memblk][first_layer][stride_unit+page_addr_base]-prefix_for_lastLayer_verification,
					msgPass_buffer[rel_bm_col][memblk][last_layer+1][stride_unit+page_addr_base]
				).any()
			):
				print(f"(Inconsistency) Layer-{layer}, Relative Base-Matrix Column_{rel_bm_col} w/ base region_{rel_bm_col_cnt}, Stride unit {stride_unit}, memblk_{memblk} -> {msgPass_buffer[rel_bm_col][memblk][first_layer][stride_unit+page_addr_base]-prefix_for_lastLayer_verification} != {msgPass_buffer[rel_bm_col][memblk][last_layer+1][stride_unit+page_addr_base]}")
				is_verification_passed=False

	return is_verification_passed

layer_num = baseMatrix_row_num
first_layer = 0
last_layer = layer_num-1			
for rel_bm_col_cnt in range(math.ceil(baseMatrix_col_num / P_c)):
	for rel_bm_col in range(P_c):
		is_verification_passed = msgPass_buffer_dump_verification(
			layer_num=layer_num,
			first_layer=first_layer,
			last_layer=last_layer,
			rel_bm_col=rel_bm_col,
			rel_bm_col_cnt=rel_bm_col_cnt
		)

		if(is_verification_passed==True):
			print(f"Relative base-matrix column {rel_bm_col} with rel_bm_col_cnt-{rel_bm_col_cnt}: Passed")
		else:
			print(f"Relative base-matrix column {rel_bm_col} with rel_bm_col_cnt-{rel_bm_col_cnt}: Failed")

