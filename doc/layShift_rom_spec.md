# Emulation of Layer-Shift ROM

The Layer-Shift ROM operates as the control memory network responsible for driving the dynamic cyclic-shift operations within the Level-1 Barrel Shifters (L1BS) and orchestrating the destination addresses for the subsequent message-pass buffer write-back routing[cite: 4]. 

A critical architectural premise of this design is that the cyclic-shift amounts across the $W^{s}$ relative row chunks within the same stride unit are not necessarily identical[cite: 4]. The emulation results demonstrate that shift amounts can vary across relative row chunks (e.g., `[45, 46, 46, 46, 46]`) for a given stride unit[cite: 4]. Consequently, independent control signaling must be provisioned for each L1BS unit[cite: 4].

## Memory Region Specifications

To support the parallel execution of the message-passing algorithm across $P_c$ base-matrix columns and $W^s$ relative row chunks, the Layer-Shift ROM is partitioned into independent physical memory blocks[cite: 4]. 

### 1. Quantity of ROM Blocks
The architecture deploys an independent Layer-Shift ROM block for every parallel row chunk across every parallel base-matrix column, yielding a total of $P_c \times W^s$ independent ROM blocks[cite: 4].

### 2. Memory Depth
The memory space inside each ROM block is subdivided into regions corresponding to the decoding layers[cite: 4]. For a given layer, the ROM must store shift and routing instructions for all stride units across all sequentially processed base-matrix columns assigned to that $P_c$ hardware lane[cite: 4].
*   **Depth per Decoding Layer:**
    $$D_{\text{layer}} = \left\lceil \frac{N_{\text{col}}}{P_c} \right\rceil \times N_{\text{strd}}$$
    *(where $N_{\text{col}}$ is the number of base-matrix columns, and $N_{\text{strd}} = \lceil Z / (P_r \cdot W^s) \rceil$ is the number of stride units[cite: 4].)*
*   **Total Depth per ROM Block:**
    $$D_{\text{total}} = N_{\text{row}} \times \left\lceil \frac{N_{\text{col}}}{P_c} \right\rceil \times N_{\text{strd}}$$
    *(where $N_{\text{row}}$ is the number of base-matrix rows[cite: 4].)*

### 3. Page Width (Word Size)
Each memory address (page) within the ROM block must store three distinct control parameters[cite: 4]:
*   **L1BS Cyclic-Shift Amount:** Determines the shift factor for the row chunk[cite: 4].
*   **Target Stride Unit:** Indicates the write-back stride unit index[cite: 4].
*   **Target Relative Row Chunk:** Indicates the write-back relative row chunk index[cite: 4].

---

## Algorithm: Generation of L1BS Shift Amounts and Routing Addresses

The following procedural flow formalizes the generation of the ROM contents, which maps absolute parity-check shifts into relative hardware-mapped execution coordinates[cite: 4].

**Input:** 
*   Base-matrix dimensions: $N_{\text{row}}, N_{\text{col}}$[cite: 4]
*   Cyclic-shift matrix: $S(i, j)$[cite: 4]
*   Architecture parameters: $Z, P_r, P_c, W^s, N_{rc}, N_{\text{strd}}$[cite: 4]

**Procedure:**

1.  **Iterate over Submatrices:**
    For each decoding layer $i \in \{0, \dots, N_{\text{row}}-1\}$ and each base-matrix column $j \in \{0, \dots, N_{\text{col}}-1\}$[cite: 4]:
    
    *   Determine the relative parallel column index: $c_{\text{rel}} = j \bmod P_c$[cite: 4].
    *   Determine the sequential execution base address: $A_{\text{base}} = \lfloor j / P_c \rfloor \times N_{\text{strd}}$[cite: 4].
    *   Identify the next decoding layer: $i_{\text{next}} = (i + 1) \bmod N_{\text{row}}$[cite: 4].

2.  **Iterate over Absolute Row Chunks:**
    For each absolute row chunk index $r_{\text{abs}} \in \{0, \dots, N_{rc}-1\}$[cite: 4]:
    
    *   **Determine Current and Next Submatrix Coordinates:**
        Let $e$ be a representative message index for the row chunk $r_{\text{abs}}$[cite: 4].
        Extract the current page and word addresses based on the current shift $S(i,j)$[cite: 4]:
        $$p_0 = \text{page\_addr}(e, S(i, j))$$
        $$w_0 = \text{word\_addr}(e, S(i, j))$$
        Extract the next page and word addresses based on the subsequent layer's shift $S(i_{\text{next}}, j)$[cite: 4]:
        $$p_1 = \text{page\_addr}(e, S(i_{\text{next}}, j))$$
        $$w_1 = \text{word\_addr}(e, S(i_{\text{next}}, j))$$
        
    *   **Calculate L1BS Cyclic-Shift Amount:**
        Calculate the required independent shift amount to align the messages for the next layer[cite: 4]:
        $$\hat{S}_{r_{\text{abs}}} = \begin{cases} P_r + w_1 - w_0, & \text{if } w_1 < w_0 \\ w_1 - w_0, & \text{otherwise} \end{cases}$$
    
    *   **Map to Relative Hardware Coordinates:**
        Convert the current absolute page address $p_0$ into hardware execution indices[cite: 4]:
        *   Current relative row chunk: $r_{\text{rel}}^{\text{cur}} = p_0 \bmod W^s$[cite: 4]
        *   Current stride unit: $u^{\text{cur}} = \lfloor p_0 / W^s \rfloor$[cite: 4]
        
        Convert the next absolute page address $p_1$ into target write-back indices[cite: 4]:
        *   Target relative row chunk: $r_{\text{rel}}^{\text{next}} = p_1 \bmod W^s$[cite: 4]
        *   Target stride unit: $u^{\text{next}} = \lfloor p_1 / W^s \rfloor$[cite: 4]
    
3.  **Populate ROM Entries:**
    Assign the computed control variables to the independent ROM memory arrays at the specific execution index $u^{\text{cur}} + A_{\text{base}}$[cite: 4]:
    
    *   **Shift Control Update:**
        $$\text{layShift\_rom}[c_{\text{rel}}][r_{\text{rel}}^{\text{cur}}][i][u^{\text{cur}} + A_{\text{base}}] = \hat{S}_{r_{\text{abs}}}$$
[cite: 4]
    *   **Write-Back Address Update:**
        $$\text{page\_waddr\_vec}[c_{\text{rel}}][r_{\text{rel}}^{\text{cur}}][i][u^{\text{cur}} + A_{\text{base}}].\text{stride\_unit} = u^{\text{next}}$$
[cite: 4]
        $$\text{page\_waddr\_vec}[c_{\text{rel}}][r_{\text{rel}}^{\text{cur}}][i][u^{\text{cur}} + A_{\text{base}}].\text{rel\_row\_chunk} = r_{\text{rel}}^{\text{next}}$$
[cite: 4]
