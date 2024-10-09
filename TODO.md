# feb2024
1. check stuff below
2. "Take your time and THINK": variable length foward propagation in small architectures
   1. tiny convex rnns as variable complexity neuron bundles 


## PHASE 0
**Pull ENTIRE thread on L-FISTA:**
idea: do dict learning on speech spectra
* STAGE A: basic
    0. ~~Setup PyTest!~~
    1. ~~Make FISTA work (unit test/toy problem)~~
    2. ~~(A) Make L-FISTA work (unit test/toy problem)~~
    2. ~~(B) (L)ISTA (subclass FISTA or vice versa)~~
    3. Set up dataloader system on mnist+asirra. Encoder/decoder pair owned by dataset?...
                 or dict point to an encoder and a dataset....
                Define trn/tst datasets that get re-used, avoiding train/test leakage.
       1. Figure out the EMG dataset a lil... some viz for github
          1. 2D complex dict on spectrograms that are overlapped?... but then how reconstruct?
       2. unit test dataloader...
       2. atom visualizer
    4. Combine into config system. Train a linear dictionary on medium data (e.g. mnist, cifar)
    5. Train L-FISTA and show it's as good + faster than FISTA. Compare with generic neural net (CNN) encoder.
            (wallclock plots, sparsity plots: cloud/histogram;; unit-test viz suite?)
    6. Repeat 0,1 for SALSA class (unit-test). 2,3 should follow automatedly.
    7. Start blog / readme /colab /notebook about this.
* STAGE B: more interesting example
    8. Try more interesting dataset
        1. 1D on financial time series?
        2. Speech separation as goal?...
        3. Generalize to 2D dictionaries?
    9. Update blog (maybe separate blog)
* STAGE C: beyond backprop
   10. Encoder subclass for saving + optimizing codes (unit-test)
   11. Alt-min style: joint dict/LFista coolness (...or is that implicit?)
   12. Update blog (maybe separate blog)

## PHASE 1:
**Make it public-ready:**
    
* STAGE A: dual-dictionary / MCA
  0. FISTA and SALSA MCA  (unit test/toy problem)
  1. Dual-dictionary class + dataloader encapsulation framework (unit test)
  2. LFISTA and LSALSA-MCA  (unit test/toy problem)
  3. LFISTA and LSALSA-MCA  (MNIST + ASIRRA)
      1. re-create all plots
      2. make classifier thing... or just skip this particular part.
  4. Update blog: official LSALSA paper blog
  5. At this point, should be done recreating paper. Email Anna (and respond to rando?)
     1. visualize the analytical results somehow? Like a N=2 version?... top couple eig vals?
* STAGE B: alt-min
  6. Use framework to train deep neural net on some classification problem (unit test?)
  7. Make each layer an algo block, try realizing parallelized layer optimization
      1. spawn Alternating Minimizer for each layer?
      2. or shold each layer be an encoder and their optimize methods get called in parallel?
  8. If it works, do unit tests and some time tests
  9. write/update blog, email Anna+Irina etc...

## Phase 2:
** Crazy Ideas...*
1. joint dictionary learning for MCA
2. use linear dictionaries to guide activation (or saliency) maps of less-interpretable architectures
3. extension of beyond-backprop to multi-task/objective?

[//]: # ()
[//]: # (----------------------------------------------------)

[//]: # (OLD TODO's)

[//]: # (----------------------------------------------------)

[//]: # (phase 0:)

[//]: # (0.&#41; forget dataset generator system; don't need it for this demo.)

[//]: # (1.&#41; focus on building the trainer + config system. I.e. the one that)

[//]: # (    has a linear case and with the right config export settings, saves as)

[//]: # (    generic CSV-like format.)

[//]: # (2.&#41; Visualizer system that can load the CSV, convert to a dictionary object,)

[//]: # (    and do interesting stuff)

[//]: # (3.&#41; &#40;rename repo, then:&#41; start blog / show different results with different cost functions and cool)

[//]: # (     visualizations)

[//]: # (4.&#41; see if anyone can run it from home)

[//]: # ()
[//]: # (phase 1:)

[//]: # (0.&#41; implement beyond backprop as an option to this framework? Other comparator?)

[//]: # (1.&#41; when it's working &#40;esp if it reproduces results&#41;, share with old)

[//]: # (    coauthors; link to IBM blog post and maybe provide own explanation)

[//]: # ()
[//]: # (phase 2:)

[//]: # (0.&#41; After all that, have a different config and trainable "algo" class that)

[//]: # (    implements unrolling and unsupervised training.)

[//]: # (1.&#41; MCA / LSALSA example)

[//]: # (2.&#41; bog post for LSALSA + MCA, link to ESP preprint...)

[//]: # ()
[//]: # (At that point probably want to move on to other repos)

[//]: # (0.&#41; Full data cleaning /web scraping example using other random skills)

[//]: # (     &#40;scala, sql, hadoop&#41;)

[//]: # (1.&#41; process data using this ML repo?)

[//]: # (2.&#41; )

[//]: # ()
[//]: # ()
[//]: # (_______________________________________________________________________________)

[//]: # (&#40;*&#41; Basic Sparse Coding)

[//]: # (0.&#41; see if other unrolling github's exist for inspiration...)

[//]: # (1.&#41; Train an MNIST dictionary)

[//]: # (2.&#41; implement ISTA/FISTA/LISTA/LSALSA using this dictionary)

[//]: # (3.&#41; write train_encoder subroutine for ^^'s)

[//]: # (3.5&#41; TRY to simplify as you go... want SIMPLE demos...)

[//]: # (4.&#41; make sure everything is working for ASIRRA and MNIST &#40;at least&#41;)

[//]: # (5.&#41; get pretty pictures and everything, save things that need to be saved for christ's sake, have this all published on github)

[//]: # (6.&#41; email mirjeta, anna, faruk, apoorva, ivan?, about having it online)

[//]: # ()
[//]: # (&#40;**&#41; do MCA:)

[//]: # (7.&#41; additive dataloader)

[//]: # (8.&#41; training regime for loading, training on set number of additive mixtures, getting target codes, etc.)

[//]: # (9.&#41; generlize LSALSA code or create separate LSALSA class that can do multiple dictionaries)

[//]: # (10.&#41; get it all running together hopefully in simplest demo possible, get to the point of making pretty pictures automatically)

[//]: # (11.&#41; try to actually recreate the experiments &#40;and plots&#41; from our paper)

[//]: # (12.&#41; email everyone)

[//]: # ()
[//]: # (&#40;***&#41; next steps:)

[//]: # (zz&#41; Do a 1D finance example!)

[//]: # (a&#41; Do a 2D example, visualize loss landscape of learned cost function)

[//]: # (b&#41; lp/lq with Mirjeta, krylov // new iterative algorithms)

[//]: # (c&#41; convolutional dictionaries, datasets-- audio?)

[//]: # (d&#41; SONAR example from SACLANT? )

[//]: # (e&#41; capability for using "smart dict's" like STFT...? basically subclass of dictionary called smart-dict or something)