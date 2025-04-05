For $i = 0, 1, \cdots, d_{v}-1$ and $j = 0, 1, \cdots d_{c}-1$, the shift control signal for a circular shifter to perform the message passing at $i$ th layer of $j$ th submatrix column is

```math
    N_{fg} = P_{r}
```

```math
    N_{strd} = \lceil Z / N_{fg} \rceil
```

```math
    \hat{S}_{cur} = S(i, j) \pmod{N_{strd}}
```

```math
    \hat{S}_{next} = S(i+1, j) \pmod{N_{strd}}
```

```math
    \hat{S}_(i, j) = 
    \begin{cases}
          \hat{S}_{next} - \hat{S}_{cur}, & \text{if}\ \hat{S}_{next} \ge \hat{S}_{cur} \\
          N_{strd} + \hat{S}_{next} - \hat{S}_{cur}, & \text{otherwise}
    \end{cases}
```

where $N_{fg}$ and $N_{strd}$ denote the numbers of stride fingers and stride sets, respectively The operation, $i+1:=i+1 \pmod{d_{v}}$. It implies that the message passing for all row chunks in submatrix $B(i, j)$  apply the identical shift control signal $\hat{S}_(i, j)$.

Let $y^{t}_{i, j}$ and $\hat{y}^{t}_{i, j}$ denote the extrinsic messages before and after message passing within $B(i, j)$, respectively; moreover, the $t$ indicates the message index across the columns of $B(i, j)$, i.e. $\forall t \in \{0, 1, \cdots, Z-1\}$. After message passing, all shifted messages are stored in the message-pass buffer at a certain bit field of a memory page. Such that the node process units can start processing the first row chunk of $(i+1)$ th layer, and read the shifted messages out from the $(0 + \text{offset\_addr})$th memory page. To recall that each memory page consists of $P_{r}$ number of $Q$-bit word data which represent shifted messages. Upon the design of message passing and message-pass buffer above, the addresses of the memory page and word region for every extrinsic message $y^{t}_{i, j}$ after message passing, are calculated by

```math
    I^{new}_{col_0} = Z-S(i, j)
```

```math
    I^{new}_{col_t} = (I^{new}_{col_0}+t) \pmod{Z}
```

```math
    word\_addr(t, i, j) =\lfloor I^{new}_{col_t} / N_{strd} \rfloor
```

```math
    page\_addr(t, i, j) = I^{new}_{col_t} \pmod{ N_{strd}}
```

From the parity-check matrix perspective, the word addresses and page addresses represent the indices of the stride fingers and stride sets, respectively.