import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List
import gc
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class MsgPassBufferBlock:
    N_strd: int
    depth_per_strd: int
    P_r: int

    msgPass_buffer_page_dtype = np.dtype([
        ('stride_unit', np.uint32),
        ('rel_row_chunk', np.uint32),
        ('value', np.uint32)
    ])

    def __post_init__(self):
        # 3D list structure: data[N_strd][depth_per_strd][P_r]
        self.data = [
            [
                [
                    np.zeros((), dtype=self.msgPass_buffer_page_dtype)
                    for _ in range(self.P_r)
                ]
                for _ in range(self.depth_per_strd)
            ]
            for _ in range(self.N_strd)
        ]


class msgPass_buffer_ctrl:

    I_row_t = None

    def __init__(
        self,
        msgPass_sched: str,
        baseMatrix_row_num: int,
        baseMatrix_col_num: int,
        Z: int,
        N_rc: int,
        W_s: int,
        P_c: int,
        P_r: int,
        # Fix 1: Correct 2D list type annotation
        msgPass_buffer_wrapper: list[list[MsgPassBufferBlock]]
    ):
        self.msgPass_sched = msgPass_sched
        self.baseMatrix_row_num = baseMatrix_row_num
        self.baseMatrix_col_num = baseMatrix_col_num
        self.Z = Z # Lifting degree of the underlying QC-LDPC code

        self.N_fg = P_r
        self.N_rc = math.ceil(Z / self.N_fg)
        self.W_s = W_s
        self.N_strd = math.ceil(Z / (P_r * W_s))
        self.P_c = P_c
        self.P_r = P_r

        # Reference pointer to the environment's 2D buffer wrapper
        self.msgPass_buffer_wrapper = msgPass_buffer_wrapper

        # Layer-Shift ROM
        layerSh_rom_page_dtype_1 = np.dtype([
            ('stride_unit', np.int32),
            ('rel_row_chunk', np.int32)
        ])

        self.memBlk_page_depth = math.ceil(self.baseMatrix_col_num / self.P_c) * self.N_strd

        self.layShift_rom = np.zeros(
            shape=(
                self.P_c,
                self.W_s,
                self.baseMatrix_row_num,
                self.memBlk_page_depth
            ),
            dtype=np.int32
        )

        self.page_waddr_vec = np.zeros(
            shape=(
                self.P_c,
                self.W_s,
                self.baseMatrix_row_num,
                self.memBlk_page_depth
            ),
            dtype=layerSh_rom_page_dtype_1
        )

    def abs_row_chunk_idx_gen(self, t: int) -> int:
        """Calculates the absolute row-chunk index I^{arc}_{t, i, j}."""
        if t >= self.Z:
            raise ValueError(f"The t: {t} >= Z: {self.Z}")

        if self.msgPass_sched == "stride_sched":
            I_arc = t % self.N_rc
            return I_arc
        else:
            raise ValueError("Unsupported message-passing scheduling scheme: " + self.msgPass_sched)

    def strd_idx_gen(self, abs_rc_idx: int) -> int:
        """Calculates the stride-unit index I^{strd}_{t, i, j}."""
        return math.floor(abs_rc_idx / self.W_s)

    def rel_row_chunk_idx_gen(self, abs_rc_idx: int) -> int:
        """Calculates the relative row-chunk index I^{rrc}_{t, i, j}."""
        return abs_rc_idx % self.W_s

    def word_addr_gen(self, t: int) -> int:
        """Calculates the intra-chunk offset / word address."""
        if t >= self.Z:
            raise ValueError(f"The t: {t} >= Z: {self.Z}")

        if self.msgPass_sched == "stride_sched":
            return math.floor(t / self.N_rc)
        else:
            raise ValueError("Unsupported message-passing scheduling scheme: " + self.msgPass_sched)

    # Write-back of the permuted messages
    def msgPass_buffer_permMsg_write(
        self,
        compMsg_vec: List[int],
        rel_bm_col: int,           # Relative base-matrix column index in range [0, P_c - 1]
        region_addr_base: int      # Base address of target region inside depth_per_strd (e.g., 0 for compV2C, 2 for permV2C)
    ):
        for msg_idx, msg_val in enumerate(compMsg_vec):
            abs_rc_idx = self.abs_row_chunk_idx_gen(t=msg_idx)
            stride_unit_idx = self.strd_idx_gen(abs_rc_idx=abs_rc_idx)
            rrc_idx = self.rel_row_chunk_idx_gen(abs_rc_idx=abs_rc_idx)
            w_addr = self.word_addr_gen(t=msg_idx)

            # Target the specific MsgPassBufferBlock reference:
            # 1. Select parallel column: wrapper[rel_bm_col][blk_idx]
            # 2. Access internal .data attribute: data[rrc_idx][stride_unit][region][word_addr]
            buffer_block = self.msgPass_buffer_wrapper[rel_bm_col][rrc_idx]
            
            # Write directly to the structured numpy element
            #print(f"abs_rc_idx: {abs_rc_idx}, rrc_idx: {rrc_idx}, stride_unit_idx:{stride_unit_idx}, region_addr_base: {region_addr_base}, w_addr: {w_addr}")
            buffer_block.data[stride_unit_idx][region_addr_base][w_addr]['stride_unit'] = stride_unit_idx
            buffer_block.data[stride_unit_idx][region_addr_base][w_addr]['rel_row_chunk'] = rrc_idx
            buffer_block.data[stride_unit_idx][region_addr_base][w_addr]['value'] = msg_val

    #---------------------------------------------------------------------------------------------------
    # For verification only
    #---------------------------------------------------------------------------------------------------
    def msgPass_buffer_dump(
            self, 
            rel_bm_col: int  # Relative base-matrix column index in range [0, P_c - 1]
        ):
            buffer_wrapper = self.msgPass_buffer_wrapper[rel_bm_col]
            print(f"\nRelative Base-Matrix Column: {rel_bm_col}")
            print("-" * 75)
            # Evaluates self.P_r in the header string
            header_word = f"word[{self.P_r - 1}:0]"
            print(f"{'Abs RC':<8} {'Rel RC':<8} {'Stride Unit':<12} {'Region':<8} {header_word}")
            print("-" * 75)

            for rel_rrc in range(self.W_s):
                block = buffer_wrapper[rel_rrc]
                for stride_unit in range(self.N_strd):
                    abs_rrc = stride_unit * self.W_s + rel_rrc

                    for region_idx, region_data in enumerate(block.data[stride_unit]):
                        # Cast NumPy scalar to int and slice [::-1] to align MSB (P_r-1) on the left
                        word_vec = [int(item['value']) for item in region_data][::-1]
                        print(
                            f"{abs_rrc:<8} {rel_rrc:<8} {stride_unit:<12} "
                            f"{region_idx:<8} {word_vec}"
                        )

class msgPass_env:

    def __init__(
        self,
        msgPass_sched: str,
        baseMatrix_row_num: int,
        baseMatrix_col_num: int,
        Z: int,
        W_s: int,
        P_c: int,
        P_r: int,
        depth_per_strd: int
    ):

        self.msgPass_sched = msgPass_sched
        self.baseMatrix_row_num = baseMatrix_row_num
        self.baseMatrix_col_num = baseMatrix_col_num
        self.Z = Z
        self.W_s = W_s
        self.P_c = P_c
        self.P_r = P_r
        self.depth_per_strd = depth_per_strd

        self.N_fg = P_r
        self.N_rc = math.ceil(Z / self.N_fg)
        self.N_strd = math.ceil(Z / (P_r * W_s))

        # Instantiate 2D array of MsgPassBufferBlock with Shape = (P_c, W_s)
        self.msgPass_buffer_wrapper = [
            [
                MsgPassBufferBlock(
                    N_strd=self.N_strd,
                    depth_per_strd=self.depth_per_strd,
                    P_r=self.P_r
                )
                for _ in range(self.W_s)
            ]
            for _ in range(self.P_c)
        ]

        # Pass the reference to msgPass_buffer_ctrl
        self.msgPass_buffer_ctrl = msgPass_buffer_ctrl(
            msgPass_sched=self.msgPass_sched,
            baseMatrix_row_num=self.baseMatrix_row_num,
            baseMatrix_col_num=self.baseMatrix_col_num,
            Z=self.Z,
            N_rc=self.N_rc,
            W_s=self.W_s,
            P_c=self.P_c,
            P_r=self.P_r,
            msgPass_buffer_wrapper=self.msgPass_buffer_wrapper
        )
