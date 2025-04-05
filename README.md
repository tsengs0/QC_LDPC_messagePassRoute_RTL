### Project Overview

This project provides the design and impelemtation of message-pass routing network for QC-LDPC decoding, where the underlying decoder architecture is assumed to be a **partially-parallel** configuration that only a fraction of check node and variable node processes mapped from each submatrix are handled concurrently. The figure below illustrates the block diagram of the proposed decoder architecture where the message-pass routing network consists of three primary building blocks with grey colour. Notably, the so-called **paritally-parallel** configuring the parallelism of one submatrix is based on the `stirde schedule` in [SS22].

![msgPass_route_blockDiagram](figures/msgPass_route_blockDiagram.png)

Add the block diagram of the message-pass routing network

---

### Development Plan

Please refer to the detail here: [development_plan.md](doc/development_plan.md)
 
---

| Type  | Name | Description | Design Rule | Declaration/Definition |
| :---   | :--- | :--- | :--- | :--- |
| 1 |  |  |  | |
| 2 |  |  | 

---

References

- [SS22] Lee, Seongjin, et al. "Multi-mode QC-LDPC decoding architecture with novel memory access scheduling for 5G new-radio standard." IEEE Transactions on Circuits and Systems I: Regular Papers 69.5 (2022): 2035-2048.
