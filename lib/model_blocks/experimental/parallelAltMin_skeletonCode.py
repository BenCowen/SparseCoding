# -*- coding: utf-8 -*-
"""
Parallelized Altmin skeleton code:
    
@author: BenJammin
"""

    #
    # for batch in dataloader:
    #
    #     codes = coder.inference(batch)
    #
    #     # not parallel:
    #     for code in codes:
    #         optimize_codes
    #
    #     # parallel"
    #     parallel_forloop n :
    #         optimize_encoder(codes[n-1], codes[n], layer[n])
    #
    #     # last layer:
    #     optimize_decoder
    #
    # def optimize_codes(prev_codes, codes, next_codes, encoder, decoder)
    #
    #     # update codes
    #     for it in range(iters):
    #
    #         loss = LOSS(codes,      activation(encoder(prev_codes))) +
    #                LOSS(next_codes,      decoder(activation(codes)))
    #
    #        code_optimizer.step()
    #
    # def optimize_encoder:
    #     loss = LOSS(next_codes, encoder(activation(prev_codes)))
    #
    # def optimizer_decoder:
    #     loss = LOSS(labels, decoder(activation(prev_codes)))
    #                 # Step 0: Code Inference
    #                 # i.e. encode the batch (forward pass)
    #                 codes = coder.encode(batch, A)
    #
    #                 # Step 1: Code Optimization
    #                 opt_codes = coder.optimize_codes(codes)
    #
    #                 # Step 2: Encoder Optimization
    #                 # (backward pass w.r.t. inference)
    #                 coder.optimize_encoder(opt_codes)
    #
    #             # Step 3: Decoder Optimization
    #             coder.optimize_decoder(opt_codes)
    #