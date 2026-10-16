---
title: "Wavelet Front-Ends and Geometric Targets for Road Segmentation"
subtitle: "A controlled study on Massachusetts Roads and DeepGlobe — architecture, equations, metrics, results and comparison with published work"
date: "October 2026"
---

# Summary

This report tests the draft paper's proposal, a single-level Haar discrete wavelet transform (DWT) in front of a ResNeXt-50 road segmentation network, under controlled conditions. It also reports a second idea that came out of the study: supervising the network with the road's centre-line and width.

**Setup.** Eight models were trained on two public benchmarks with 3 seeds each, giving 48 runs. All runs used one training recipe and one evaluation protocol. The ResNeXt variants (R0–R3, GA, GB) share the same encoder, decoder, data stream and seeds and differ only in the component under test. The analysis, including the decision rule, was committed to the code repository before any test score was read.

**Main findings.**

1. **The DWT front-end (the draft paper's method, R1) gives no detectable improvement.** The pre-specified primary comparison R1 − R0 in per-image test IoU is +0.0004 (95% CI −0.0040 to +0.0060) on Massachusetts Roads and +0.0017 (−0.0018 to +0.0059) on DeepGlobe. Section 6.3 shows why: a DWT followed by a convolution is a re-parameterised strided convolution, so it adds no information that a pretrained first layer lacks.
2. **Learned wavelet pooling throughout the network (R2)** is ahead of R0 on DeepGlobe on every metric (pooled IoU +0.0052). Under the pre-specified rule the result is *inconclusive* at three seeds, and R2 shows no gain on Massachusetts.
3. **Centre-line + width auxiliary supervision (GA) is the one confirmed improvement.** On DeepGlobe it raises per-image IoU by +0.0073 (95% CI +0.0033 to +0.0118, seed-paired *t* +0.0011 to +0.0135). This contrast was registered as confirmatory before the DeepGlobe result was read. GA reaches **IoU 0.6858, mIoU 0.8352, F1 0.8136, precision 0.8087 and accuracy 0.9851**. The gain comes from fewer false positives (precision +1.1 points) at unchanged recall. On Massachusetts GA is +0.0024 (inconclusive; two more seeds are training).
4. **Against published results.** On DeepGlobe, GA's road IoU (0.6858) is higher than every published result we compared against except DSFC-Net (0.6868), which it matches within 0.001. That includes D-LinkNet (challenge winner, 0.6466), NL-LinkNet (0.650), RoadCNN (0.6761), Batra et al. (0.6721), Seg-Road (0.6720), SegFormer (0.6742) and URoadNet (0.6777). Its mIoU (0.8352) is higher than Seg-Road's (0.8206). On Massachusetts our models (IoU 0.652–0.663) beat the baselines that those papers re-ran (0.601–0.646) but are below the newest published methods (0.675–0.684). Each paper uses its own split and protocol, so these comparisons are indicative only (Section 11).

**Metric definitions.** F3 and F4 are the F-β scores with β = 3 and β = 4, which weight recall 9 and 16 times as heavily as precision. mIoU is the mean of the road IoU and the background IoU. All are defined in Section 8.

# Introduction and research question

Road extraction from aerial and satellite images is a binary segmentation problem with two characteristic difficulties: roads are thin (2–20 pixels wide) and they are often partly hidden by trees, shadows and vehicles. Small pixel errors can break the connectivity of a road network.

The draft paper proposes decomposing each colour channel of the input image with a single-level 2-D Haar DWT, stacking the 12 resulting sub-bands, and feeding them to a ResNeXt encoder whose first convolution is widened from 3 to 12 input channels. The motivation is that the high-frequency sub-bands expose edges that help locate thin, occluded roads.

**Question.** Does the DWT front-end improve road segmentation when everything else is held fixed, and if not, is there a wavelet or shape-based change that does?

**Design.** An *ablation ladder*: each rung changes exactly one thing relative to the control R0.

| Rung | Change relative to R0 | Role |
|:--------|:-------------------------------------------------------|:----------------------|
| R0 | — (stock ImageNet ResNeXt-50 32×4d encoder + U-Net decoder) | control |
| R1 | Haar DWT at the input; stride-1 stem; exact initialisation from R0 | **draft paper's method** |
| R2 | learned Haar-basis pooling at every down-sampling, wavelet unpooling in the decoder | strong wavelet variant |
| R3 | R2 + soft-clDice topology loss | connectivity variant |
| GA | R0 + auxiliary centre-line and width heads (training signal only) | geometric supervision |
| GB | GA, but the mask is *drawn* from the predicted centre-line and width | geometric representation |
| U-Net | classic U-Net trained from scratch | external baseline |
| D-LinkNet | ResNet-34 LinkNet with dilated centre block (DeepGlobe 2018 winner) | external baseline |

Table: The ablation ladder and baselines.

# Related work (literature survey)

**Road extraction as pixel classification.** Mnih and Hinton [2] introduced the Massachusetts Roads benchmark, and the relaxed precision and recall that tolerate a few pixels of misregistration between label and image [1]. Fully convolutional networks [13] and the U-Net encoder–decoder with skip connections [6] replaced patch classifiers. The residual U-Net of Zhang et al. [14] reached a relaxed break-even of 0.9187 on Massachusetts with a quarter of U-Net's parameters. On DeepGlobe [3], D-LinkNet [7] won the 2018 challenge. It is a LinkNet with an ImageNet-pretrained ResNet-34 encoder and a centre block of cascaded dilated convolutions. NL-LinkNet [15] replaced the dilated block with non-local operations.

**Connectivity and topology.** Pixel-wise losses tolerate small gaps that break a road network. DeepRoadMapper [16] repaired gaps with graph reasoning. Mosinska et al. [17] added a perceptual topology loss. Batra et al. [18] learned road orientation jointly with segmentation. CoANet [19] used strip convolutions and a connectivity-attention module, and clDice [8] measures and optimises the overlap of soft skeletons. Connectivity is commonly evaluated with APLS [9], which compares shortest paths in graphs built from prediction and ground truth.

**Graph extraction and foundation models.** RoadTracer [20], Sat2Graph [21] and SAM-Road [22] predict the road graph directly. They are evaluated on graph metrics and are not directly comparable with raster metrics.

**Transformers and hybrid encoders.** Recent results are dominated by transformer or hybrid CNN–transformer encoders. Examples are SegFormer [23] as a generic baseline; Seg-Road [24], which pairs a transformer branch with a CNN branch and a pixel-connectivity structure; URoadNet [25], a U-Net with dual sparse attention; and DSFC-Net [26], which separates high- and low-frequency content with a Laplacian pyramid inside a spatial–frequency hybrid transformer.

**Wavelets in CNNs.** The multiresolution wavelet representation [5] has entered deep networks as pooling [27], as lossless down- and up-sampling (MWCNN [12], WaveCNet [11, 28], WaveSNet for segmentation [29]), as Haar wavelet down-sampling for segmentation [30], and as a Fourier/wavelet road-extraction module (PWFNet [31]). Anti-aliased down-sampling [10] restores the shift-invariance that strided operations destroy, which is the same mechanism several wavelet methods appeal to.

**Width and centre-line representations.** Road masks in both benchmarks are reliable about *where a road runs* but not about *where its edges are*: Massachusetts masks are rasterised centre-lines of almost constant width (median 5.3 px), and DeepGlobe widths vary between images (median 9.8 px, 5th–95th percentile 5–22 px). This motivates supervising the centre-line and width explicitly (GA/GB).

**Position of this study.** Most wavelet papers compare a wavelet-equipped network with a vanilla counterpart and report gains. We are not aware of a road-extraction study that (i) keeps every pretrained weight and initialises the wavelet variant to be *functionally identical* to the baseline, (ii) matches data stream, recipe and seeds across variants, and (iii) fixes the primary comparison and decision rule before seeing test results.

# Datasets

| Dataset | Tile size | Resolution | Train / val / test | Road pixels | No-data masking |
|:----------------------|:-----------:|:----------:|:------------------:|:----------:|:----------:|
| Massachusetts Roads | 1500 × 1500 | 1.0 m/px | 1108 / 14 / 49 (official) | ≈ 5% of valid | yes |
| DeepGlobe Road Extraction | 1024 × 1024 | 0.5 m/px | 4358 / 933 / 935 (seed 1337) | ≈ 3.6% | not needed |

Table: Datasets used.

**Massachusetts Roads** [1]: aerial RGB imagery of Massachusetts with masks rasterised from OpenStreetMap centre-lines. Official split.

- Original source (used here): <https://www.cs.toronto.edu/~vmnih/data/>
- Kaggle mirror: <https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset>

Many Massachusetts tiles contain large blank white areas with no imagery (11% of pixels on average in a 40-image sample, up to 79%), and the labels draw roads through them. A pixel is treated as *no-data* if all three channels are ≥ 250 **and** it lies in a 4-connected white region of at least 2500 pixels. The resulting validity mask *V* is excluded from the loss, crop sampling, model selection and every metric.

**DeepGlobe Road Extraction** [3]: satellite RGB imagery (Thailand, Indonesia, India). Only the 6226 training tiles were released with labels, so they were re-split by image 70/15/15% with a fixed seed. The test split is pinned by a SHA-256 fingerprint of its tile ids.

- Challenge page: <http://deepglobe.org/>
- Kaggle release: <https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset>
- Mirror used here: <https://huggingface.co/datasets/lammtfkday/deepglobe-roads> (counts, tile size and resolution match the original release; pixel identity was not verified).

**No resizing.** Images are never resized, because roads are only 2–4 px wide at their narrowest. Training uses 512 × 512 crops, and evaluation tiles each full image and stitches the predictions back together.

# Architecture of the entire pipeline

![**Figure 1. End-to-end pipeline.** (1) Data: datasets, splits, no-data mask, in-memory preload. (2) Training: crop sampler, augmentation, network (Figure 2), loss, optimiser, repeated per epoch. (3) Model selection on validation, tiled test inference, binarisation, metrics and statistics.](fig/pipeline.png){width=6.5in}

The pipeline has three stages.

**Stage 1: Data.**

1. *Load* the dataset and its fixed split (official for Massachusetts; seeded, fingerprinted re-split for DeepGlobe).
2. *No-data mask* *V* (Massachusetts only), applied everywhere downstream.
3. *Preload* the decoded images once into shared memory so that data-loader workers do not re-decode them.

**Stage 2: Training (identical for every model).**

1. *Crop sampler.* Random 512 × 512 crops, 8 × (number of training images) per epoch. With probability 0.75 a crop is re-drawn until it contains road (up to 12 tries), and crops with less than 50% valid pixels are rejected. All randomness comes from one seeded generator, so for a given seed **every model sees the identical crop stream**.
2. *Augmentation.* Horizontal and vertical flips and 90° rotations. With probability 0.5, an affine warp (±15°, scale 0.9–1.1). With probability 0.5, brightness/contrast jitter in [0.9, 1.1]. ImageNet normalisation is applied last.
3. *Network* (Section 6, Figure 2): ResNeXt-50 encoder + U-Net decoder, with the rung-specific change.
4. *Loss* in fp32, masked by *V* (Section 7).
5. *Optimiser:* AdamW with a warm-up + cosine learning-rate schedule, bf16 autocast, batch 16 and gradient clipping at 1.0. Checkpoints are written every epoch, so runs resume after pre-emption.

**Stage 3: Selection, evaluation and statistics.**

1. *Model selection* on the validation split only: the best epoch by pooled IoU, then the decision threshold τ\* ∈ {0.20, 0.25, …, 0.80} that maximises validation IoU.
2. *Test inference:* 512 × 512 tiles at stride 256, overlapping probabilities averaged, stitched to full resolution.
3. *Binarise* at τ\*. Each run's test split is scored **exactly once**.
4. *Metrics* (Section 8): accuracy, precision, recall, F1, F3, F4, IoU, mIoU, relaxed F1, clDice, APLS.
5. *Statistics* (Section 9): two-level bootstrap and seed-paired *t* interval with a pre-specified decision rule.

# Network architecture

![**Figure 2. Network.** ResNeXt-50 32×4d encoder (blue), U-Net decoder (orange), skip connections (dashed), and output heads (green). R0–R3 use the mask head only; GA adds centre and width heads as auxiliary outputs; GB draws the mask from them with the differentiable Render operator.](fig/network.png){width=6.5in}

## Haar wavelet transform as fixed convolutions

With low- and high-pass filters $\ell = \tfrac{1}{\sqrt2}(1,1)$ and $h = \tfrac{1}{\sqrt2}(1,-1)$, the four 2 × 2 analysis kernels are the outer products

$$\kappa_{LL} = \ell^\top\ell,\qquad \kappa_{LH} = h^\top\ell,\qquad \kappa_{HL} = \ell^\top h,\qquad \kappa_{HH} = h^\top h .$$

For a 2 × 2 pixel block with values $a, b$ (top row) and $c, d$ (bottom row):

$$\text{LL} = \frac{a+b+c+d}{2},\quad \text{LH} = \frac{a+b-c-d}{2},\quad \text{HL} = \frac{a-b+c-d}{2},\quad \text{HH} = \frac{a-b-c+d}{2}.$$

The four kernels form an orthonormal basis, so applied channel-wise with stride 2,

$$\text{DWT}:\ \mathbb{R}^{C\times H\times W}\ \to\ \mathbb{R}^{4C\times \frac{H}{2}\times\frac{W}{2}}$$

is a grouped convolution with frozen weights, and its inverse (IDWT, the transposed convolution with the same kernels) is exact: $\text{IDWT}(\text{DWT}(x)) = x$. Both run on the GPU inside the network.

## Encoder: ResNeXt-50 (32 × 4d)

Each bottleneck block aggregates $C = 32$ parallel transformations (cardinality) [4]:

$$y = x + \sum_{i=1}^{32} \mathcal{T}_i(x),$$

implemented as a 1 × 1 reduction, a 3 × 3 convolution with 32 groups and a 1 × 1 expansion. For a 512 × 512 input, the encoder passes five feature maps to the decoder:

| Feature | Produced by | Resolution | Channels |
|:--|:--|:--:|:--:|
| $f_0$ | stem | 256 × 256 | 64 |
| $f_1$ | layer1 | 128 × 128 | 256 |
| $f_2$ | layer2 | 64 × 64 | 512 |
| $f_3$ | layer3 | 32 × 32 | 1024 |
| $f_4$ | layer4 (bottleneck) | 16 × 16 | 2048 |

Table: Encoder feature maps.

## R1 — DWT-input stem with an exact initialisation (the draft paper's method)

**Stride parity.** The DWT already halves the resolution, so R1's stem convolution has stride 1. All later feature maps, skips and the decoder therefore have the same shapes as in R0:

$$\text{R0:}\ \ x \in \mathbb{R}^{3\times512^2} \xrightarrow{\ \text{conv } 7\times7,\ s=2\ } \mathbb{R}^{64\times256^2},\qquad
\text{R1:}\ \ x \xrightarrow{\ \text{DWT}\ } \mathbb{R}^{12\times256^2} \xrightarrow{\ \text{conv } 7\times7,\ s=1\ } \mathbb{R}^{64\times256^2}.$$

**Exact initialisation (Proposition).** Let $W \in \mathbb{R}^{O\times3\times7\times7}$ be R0's pretrained stem kernel (stride 2, padding 3). Let $K$ be $W$ zero-padded by one row at the top and one column at the left (8 × 8), and $G = \text{DWT}(K) \in \mathbb{R}^{O\times12\times4\times4}$. Setting $W'[:,:,1{:}5,1{:}5] = G$ (zero elsewhere) gives, for every input with even height and width,

$$\text{conv}_{s=1,\,p=3}\big(\text{DWT}(x);\,W'\big) \;=\; \text{conv}_{s=2,\,p=3}\big(x;\,W\big).$$

*Why.* Within each 2 × 2 block, the inner product of kernel and pixels equals the inner product of their Haar coefficients, because the Haar basis is orthonormal. A stride-2 7 × 7 convolution therefore becomes a stride-1 4 × 4 convolution on the coefficient map.

**Consequence.** R1 is the *same function* as the pretrained R0 at step 0 (maximum logit difference measured at 1.5 × 10⁻⁸), and can differ only by learning. Conversely, "DWT then convolution" is just a re-parameterised strided convolution with a larger (14 × 14) footprint: **the DWT adds no information**. This explains the null result in Section 10.

## R2 — learned Haar-basis pooling and unpooling

**WaveletPool** replaces every stride-2 operation of the encoder (stem max-pool and the stride-2 convolutions and shortcuts of layers 2–4). It is placed in front of the original pretrained layer, whose stride is set to 1. For sub-bands $D = \text{DWT}(x)$,

$$\text{WP}(x)_c = \sum_{k=0}^{3} \omega_{c,k}\, D_{4c+k},\qquad \omega \in \mathbb{R}^{C\times4}.$$

Initialising $\omega_c = (\tfrac12, 0, 0, 0)$ gives 2 × 2 average pooling. Initialising $\omega_c = (\tfrac12,\tfrac12,\tfrac12,\tfrac12)$ selects the top-left pixel, which is exactly what a stride-2 1 × 1 shortcut reads, so that replacement is exact.

**WaveletUnpool** replaces bilinear up-sampling in the decoder:

$$\text{WU}(x) = \text{IDWT}\big([\,\omega_{c,k}\,x_c\,]_{c,k}\big),\qquad \omega_c = (2,0,0,0) \text{ at initialisation (nearest-neighbour)}.$$

## Decoder and output head

A U-Net decoder [6] with explicit skips:

$$d_\ell = \phi_\ell\big([\,\text{Up}(d_{\ell-1})\ ;\ f_{4-\ell}\,]\big),\qquad d_0 = f_4,$$

where $\phi_\ell$ is two 3 × 3 Conv–BN–ReLU layers, with widths 512, 256, 128, 64, 32 at resolutions 32² … 512². A 1 × 1 convolution gives one road logit per pixel.

## GA and GB — centre-line and width heads

**Targets** are derived from each training crop's mask by image processing only:

- the **centre** target is the skeleton of the mask plus any mask pixels its discs leave uncovered;
- the **width** target at each centre pixel is the largest radius in $\{0,1,2,3,4,5,6,8,10,13,16,20\}$ whose disc fits inside the mask (12 classes).

Drawing the discs back reproduces every training mask exactly (IoU 1.000 on 60 checked images per dataset).

**GA** adds two 1 × 1 heads to R0's decoder output: a centre logit $c$ and 12 width logits $w$. They are used only as an extra training signal; the mask is still predicted by the normal mask head.

**GB** instead *draws* the mask from the centre and width predictions with a differentiable operator:

$$m = \text{Render}\big(\sigma(c)\cdot\text{softmax}(w)\big),$$

which dilates each width bin by its radius using successive 3 × 3 max-pools (an octagonal disc) and takes the pixel-wise maximum.

## Model sizes

| Model | Trainable parameters |
|:--|--:|
| R0 | 46,503,137 |
| R1 | 46,531,361 |
| R2, R3 | 46,557,985 |
| GA, GB | 46,503,566 (R0 + 429 for the two 1 × 1 heads) |
| U-Net (from scratch) | 31,037,633 |
| D-LinkNet (ResNet-34) | 31,095,137 |

Table: Parameter counts. The wavelet rungs add 0.06–0.12% to R0.

# Training

Let $z$ be the logits, $p = \sigma(z)$, $y \in \{0,1\}$ the mask and $m$ the validity mask; sums run over the pixels of a 512 × 512 crop.

**Binary cross-entropy and Dice** (R0, R1, R2, baselines):

$$\mathcal{L}_{\text{BCE}} = \frac{\sum m\,\text{bce}(z,y)}{\max(\sum m,\,1)},\qquad
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum p\,y\,m + 1}{\sum p\,m + \sum y\,m + 1},\qquad
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}.$$

**clDice (R3).** With a differentiable soft skeleton $S(\cdot)$ [8]:

$$T_{\text{prec}} = \frac{\sum S(p)\,y}{\sum S(p)},\qquad T_{\text{sens}} = \frac{\sum S(y)\,p}{\sum S(y)},\qquad
\text{clDice} = \frac{2\,T_{\text{prec}}\,T_{\text{sens}}}{T_{\text{prec}} + T_{\text{sens}}},\qquad
\mathcal{L}_{\text{R3}} = \mathcal{L} + 0.5\,(1-\text{clDice}).$$

**Geometric loss (GA, GB).** For GB the mask term uses the drawn mask:

$$\mathcal{L}_{\text{geo}} = (\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}})(\text{mask}) + 1.0\,(\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}})(\text{centre}) + 1.0\,\text{CE}(\text{width}) \ \text{over centre pixels}.$$

**Optimisation.** AdamW, $\eta_0 = 3\times10^{-4}$, weight decay $10^{-4}$, batch 16, gradient clipping 1.0, bf16 autocast (losses in fp32). Per-iteration schedule with $T_w$ warm-up steps (2 epochs) out of $T$:

$$\eta(t) = \begin{cases} \eta_0\left(0.01 + 0.99\,\dfrac{t}{T_w}\right), & t < T_w \\[6pt] \eta_{\min} + \tfrac12(\eta_0-\eta_{\min})\left(1+\cos\dfrac{\pi(t-T_w)}{T-T_w}\right), & t \ge T_w \end{cases}\qquad \eta_{\min} = 0.01\,\eta_0 .$$

Massachusetts runs for 100 epochs and DeepGlobe for 40. One recipe is used for every model; nothing was tuned per model. All runs used single NVIDIA L40S GPUs.

# Evaluation metrics

All metrics are computed on the stitched full-resolution test images, over valid pixels only, at the validation-selected threshold τ\*. Road is the positive class. With TP, FP, FN and TN counted over all test pixels (pooled):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN},\qquad
\text{Precision} = \frac{TP}{TP + FP},\qquad
\text{Recall} = \frac{TP}{TP + FN}$$

$$F_\beta = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2\,FN + FP},\qquad F_1 = \frac{2\,TP}{2\,TP + FN + FP}$$

**F3 and F4** are $F_\beta$ with β = 3 and β = 4. They weight recall β² = 9 and 16 times as heavily as precision, which suits applications where a missed road costs more than a false alarm. (We read "F3, F4" as F-β; if another meaning was intended, it can be recomputed from the same counts.)

$$\text{IoU}_{\text{road}} = \frac{TP}{TP + FP + FN},\qquad
\text{IoU}_{\text{bg}} = \frac{TN}{TN + FN + FP},\qquad
\text{mIoU} = \tfrac12\left(\text{IoU}_{\text{road}} + \text{IoU}_{\text{bg}}\right)$$

**Notes.**

- **IoU** in the tables is the road IoU pooled over the test set; "per image" averages each image's IoU.
- **Accuracy and mIoU include the background**, which covers about 95–96% of pixels. That makes them high for every model (accuracy ≈ 0.98), so road IoU and F1 are the more discriminating numbers.
- **Relaxed F1** [2] counts a predicted road pixel as correct if a true road pixel lies within ρ = 3 px, and vice versa.
- **clDice** [8] measures the overlap of skeletons.
- **APLS** [9] compares shortest-path lengths between graphs built from the predicted and true skeletons (500 node pairs per direction, 8 px snapping).

# Statistical analysis

Let $u_{a,s,i}$ be the test IoU of image $i$ for arm $a$ (baseline or comparison) and seed $s$. The effect is the mean per-image difference $\hat\Delta = \frac{1}{N}\sum_i (\bar u_{1,i} - \bar u_{0,i})$.

- **Two-level bootstrap (B = 10,000).** Each replicate re-samples seeds within each arm, then images (the same images for both arms). The percentile interval gives the 95% CI.
- **Seed-paired *t* interval.** Seeds are paired by design (same crop stream, same initial weights). With $d_s$ the per-seed mean difference over $n = 3$ seeds, the interval is $\bar d \pm t_{0.975,\,2}\, s_d/\sqrt{3}$.
- **Pre-specified decision rule.**
  - A *difference* is claimed only if **both** intervals exclude zero with the same sign.
  - If only one excludes zero, the result is *inconclusive*.
  - If neither excludes zero, there is *no detected difference*.

# Results

All runs completed without failures. Values are mean ± standard deviation over 3 seeds, in percent. **Bold** marks the best mean in each column. Model labels: R0 = ResNeXt-50 U-Net control; R1 = DWT at the input (the draft paper); R2 = learned wavelet pooling/unpooling; R3 = R2 + clDice; GA = R0 + auxiliary centre-line/width heads; GB = mask drawn from centre-line/width; U-Net trained from scratch; D-LinkNet with ResNet-34 (Table 1).

## Required metrics: accuracy, precision, recall, F1, F3, F4, IoU, mIoU

{{MAIN_deepglobe}}

Table: **DeepGlobe** test set (935 images): pooled pixel metrics (%).

{{MAIN_massachusetts}}

Table: **Massachusetts Roads** test set (49 images): pooled pixel metrics over valid pixels (%).

**Reading the tables.**

- On **DeepGlobe**, GA is best on accuracy, precision, F1, IoU and mIoU. GB and R2 are best on recall, F3 and F4: they find slightly more road at the cost of more false positives.
- On **Massachusetts** the models are within about one IoU point of one another. The from-scratch U-Net is best on most metrics, and GA is the best of the ResNeXt models.

## Topology and per-image metrics (secondary)

{{TOPO_deepglobe}}

Table: **DeepGlobe**: per-image IoU and topology metrics (%).

{{TOPO_massachusetts}}

Table: **Massachusetts Roads**: per-image IoU and topology metrics (%).

## Controlled comparisons (pre-specified analysis)

| Data | Contrast | Δ IoU | Bootstrap 95% CI | Seed-paired *t* 95% CI | Decision |
|:--------|:--------------------|-------:|:--------------:|:--------------:|:-----------------------|
| Mass. | **R1 − R0** (primary, paper 1) | +0.0004 | [−0.0040, +0.0060] | [−0.0095, +0.0104] | no detected difference |
| DG | **R1 − R0** (primary, paper 1) | +0.0017 | [−0.0018, +0.0059] | [−0.0072, +0.0106] | no detected difference |
| Mass. | R2 − R0 | −0.0009 | [−0.0060, +0.0049] | [−0.0099, +0.0081] | no detected difference |
| DG | R2 − R0 | +0.0039 | [+0.0001, +0.0084] | [−0.0059, +0.0137] | inconclusive |
| Mass. | R3 − R0 | −0.0018 | [−0.0061, +0.0020] | [−0.0052, +0.0017] | no detected difference |
| DG | R3 − R0 | −0.0027 | [−0.0069, +0.0019] | [−0.0112, +0.0058] | no detected difference |
| Mass. | **GB − R0** (primary, paper 2) | −0.0021 | [−0.0060, +0.0016] | [−0.0056, +0.0013] | no detected difference |
| DG | **GB − R0** (primary, paper 2) | −0.0017 | [−0.0059, +0.0028] | [−0.0129, +0.0095] | no detected difference |
| Mass. | GA − R0 | +0.0024 | [−0.0020, +0.0069] | [+0.0018, +0.0031] | inconclusive (seeds 3–4 running) |
| DG | **GA − R0** (confirmatory) | **+0.0073** | **[+0.0033, +0.0118]** | **[+0.0011, +0.0135]** | **difference: GA better** |
| Mass. | GB − GA | −0.0046 | [−0.0087, −0.0008] | [−0.0083, −0.0008] | difference: GB worse |
| DG | GB − GA | −0.0090 | [−0.0118, −0.0062] | [−0.0155, −0.0025] | difference: GB worse |
| Mass. | U-Net − R0 | +0.0095 | [+0.0047, +0.0151] | [+0.0068, +0.0123] | difference: U-Net better |
| DG | U-Net − R0 | −0.0332 | [−0.0381, −0.0277] | [−0.0421, −0.0243] | difference: U-Net worse |
| Mass. | D-LinkNet − R0 | −0.0043 | [−0.0114, +0.0023] | [−0.0069, −0.0017] | inconclusive |
| DG | D-LinkNet − R0 | −0.0081 | [−0.0126, −0.0033] | [−0.0175, +0.0012] | inconclusive |

Table: Paired contrasts in per-image test IoU (Mass. = Massachusetts Roads, DG = DeepGlobe). Only the rows marked primary or confirmatory were fixed in advance as hypothesis tests; the other rows are secondary and not corrected for multiplicity.

**Interpretation.**

- **The DWT front-end (R1) does not help a pretrained ResNeXt** on either dataset. The intervals exclude gains larger than about 0.6 IoU points, which is consistent with the Proposition in Section 6.3.
- **Learned wavelet pooling (R2)** may help on DeepGlobe, but three seeds are not enough to tell. If it has a benefit, robustness to shifts and corruptions is a likelier place to find it.
- **Supervising the centre-line and width (GA)** improves DeepGlobe IoU, and the improvement is confirmed. Using them to *draw* the mask (GB) is worse than using them only as supervision.

# Comparison with published results

| Method | Split / protocol | Mass. IoU | DeepGlobe IoU | Other metrics as published |
|:--------------------|:----------------------|:-------:|:----------:|:----------------------|
| D-LinkNet [7] | DeepGlobe challenge server (val / test) | — | 0.6466 / 0.6342 | — |
| NL-LinkNet [15] | DeepGlobe challenge val | — | 0.6500 | ("mIoU" as named in source) |
| RoadCNN [20], as reported in [18] | DG 4696 / 1530, full image | — | 0.6761 | — |
| Batra et al. [18] | DG 4696 / 1530, full image | — | 0.6721 | APLS 0.7312 |
| D-LinkNet, as reported in [24] | Mass 1108/14/49; DG 4696/1530; 512² crops | 0.6145 | 0.5762 | — |
| CoANet [19], as reported in [24] | as above | 0.6167 | 0.5911 | — |
| Seg-Road-l [24] | as above | 0.6838 | 0.6720 | mIoU Mass 0.8389, DG 0.8206 |
| SegFormer [23], as reported in [25] | Mass official; DG 4891/190/1145; resized to 512² | 0.6009 | 0.6742 | — |
| URoadNet [25] | as above | 0.6761 | 0.6777 | accuracy Mass 0.9706, DG 0.9753 |
| U-Net, as reported in [26] | Mass 1108/63; DG 4980/1236; patches | 0.6456 | 0.6506 | — |
| D-LinkNet, as reported in [26] | as above | 0.6443 | 0.6599 | — |
| DSFC-Net [26] | as above | 0.6750 | 0.6868 | — |
| **Ours: R0** (ResNeXt-50 U-Net) | Mass official, no-data masked; DG 4358/933/935; stitched full images; 3 seeds | 0.6562 | 0.6771 | mIoU 0.8181 / 0.8306 |
| **Ours: R1** (draft paper, DWT) | as above | 0.6554 | 0.6785 | mIoU 0.8176 / 0.8314 |
| **Ours: GA** (centre + width supervision) | as above | 0.6593 | **0.6858** | mIoU 0.8197 / **0.8352**; accuracy 0.9809 / **0.9851**; F1 0.7946 / 0.8136 |
| **Ours: U-Net** (from scratch) | as above | **0.6632** | 0.6461 | mIoU 0.8217 / 0.8141 |

Table: Published road-segmentation results and ours (values as printed in each source, converted to 0–1). Splits and protocols differ between row groups.

## Where our results are better

**DeepGlobe: road IoU.** GA (0.6858) is higher than:

- the 2018 challenge winner D-LinkNet (0.6466 val / 0.6342 test) and NL-LinkNet (0.650);
- RoadCNN (0.6761) and Batra et al. (0.6721);
- Seg-Road (0.6720), SegFormer (0.6742) and URoadNet (0.6777).

It matches DSFC-Net (0.6868; difference −0.0010). Even our plain R0 control (0.6771) is on a par with the strongest of these.

**DeepGlobe: other metrics.**

- GA's mIoU (0.8352) is higher than Seg-Road's (0.8206).
- GA's pixel accuracy (0.9851) is higher than URoadNet's (0.9753).
- On Massachusetts, our accuracy (0.9802–0.9809 for every model) is higher than URoadNet's (0.9706).

**Massachusetts: re-run baselines.** Every model we trained (IoU 0.6515–0.6632) is higher than the baselines that the recent papers re-ran: D-LinkNet 0.6145 and CoANet 0.6167 in [24], SegFormer 0.6009 in [25], and U-Net 0.6456 and D-LinkNet 0.6443 in [26]. Our pipeline therefore gets more out of standard architectures than those papers' baseline pipelines did.

## Where our results are not better

On Massachusetts, the best published methods remain ahead in road IoU: Seg-Road 0.6838, URoadNet 0.6761 and DSFC-Net 0.6750, against our 0.6632 (U-Net) and 0.6593 (GA). Seg-Road's mIoU (0.8389) is also higher than ours (0.8217). The DWT model of the draft paper (R1) is not better than its own control on either dataset.

## Caveats — why these cross-paper numbers are indicative only

- **Different test sets.** DeepGlobe's official test labels were never released, so every paper evaluates on its own split.
- **Different protocols.** Papers differ in crop size, resizing (URoadNet resizes to 512²; we never resize), whether metrics are computed on crops or on stitched full images, and whether "IoU" means road IoU or the mean with background.
- **Protocol alone moves numbers by several points.** The same architecture, D-LinkNet, is reported at 0.6466 by its authors, 0.6599 by [26] and 0.5762 by [24] on DeepGlobe; our reproduction gives 0.6655.
- **Our Massachusetts scores exclude no-data pixels**, which published numbers include.
- **Accuracy is dominated by the background** and depends on each split's road fraction, so the accuracy comparison is the weakest of all.
- **Some published F1 values are computed differently.** For one class, IoU = F1 / (2 − F1). Seg-Road's printed F1 (0.9143 with IoU 0.6720) does not satisfy this, so its F1 is evidently computed differently and we do not compare against it.

**The strongest evidence in this report is therefore the controlled comparison in Section 10.3.** Same split, same recipe, same seeds and pre-specified intervals show that GA beats its own control on DeepGlobe, and that the DWT front-end does not.

# Limitations

- **Statistical power.** Three seeds per model; the bootstrap interval can be anti-conservative when seed variance dominates. Two more Massachusetts seeds for R0 and GA, and attribution runs for GA (centre-only and width-only heads), are training at the time of writing.
- **Confounded wavelet rungs.** R2 and R3 change the encoder and decoder operators together.
- **No tuning.** One untuned recipe for every model; GA/GB loss weights (1.0, 1.0) and the radius bins were not tuned.
- **Small Massachusetts validation set.** Threshold and checkpoint selection there uses only 14 images.
- **Dataset provenance.** The DeepGlobe split is our own random split by image, and the files come from a mirror whose pixel identity with the original release was not verified.
- **Reproducibility.** bf16 and non-deterministic GPU kernels make reruns statistically, not bitwise, reproducible.

# Reproducibility and code

- **Code:** private repository <https://github.com/HarshithKoriRaj/roadseg> (branches `feat/dwt-resnext-ablation` and `feat/geo-targets`).
- **Run records:** every run directory records the resolved configuration, seed, code commit, library versions, pretrained-weight hashes, the DeepGlobe split fingerprint, training history, selected threshold and per-image test scores.
- **Tests:** the code includes a test suite, a CPU end-to-end rehearsal and a GPU self-test.
- **Launches:** all runs were launched from frozen code snapshots.

# References

1. V. Mnih. *Machine Learning for Aerial Image Labeling*. PhD thesis, University of Toronto, 2013.
2. V. Mnih and G. E. Hinton. Learning to detect roads in high-resolution aerial images. *ECCV*, 2010.
3. I. Demir et al. DeepGlobe 2018: A challenge to parse the Earth through satellite images. *CVPR Workshops*, 2018.
4. S. Xie, R. Girshick, P. Dollár, Z. Tu and K. He. Aggregated residual transformations for deep neural networks (ResNeXt). *CVPR*, 2017.
5. S. G. Mallat. A theory for multiresolution signal decomposition: the wavelet representation. *IEEE TPAMI* 11(7), 1989.
6. O. Ronneberger, P. Fischer and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 2015.
7. L. Zhou, C. Zhang and M. Wu. D-LinkNet: LinkNet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction. *CVPR Workshops*, 2018.
8. S. Shit et al. clDice — a novel topology-preserving loss function for tubular structure segmentation. *CVPR*, 2021.
9. A. Van Etten, D. Lindenbaum and T. M. Bacastow. SpaceNet: A remote sensing dataset and challenge series. arXiv:1807.01232, 2018.
10. R. Zhang. Making convolutional networks shift-invariant again. *ICML*, 2019.
11. Q. Li, L. Shen, S. Guo and Z. Lai. WaveCNet: Wavelet integrated CNNs to suppress aliasing effect for noise-robust image classification. *IEEE TIP* 30, 2021.
12. P. Liu, H. Zhang, K. Zhang, L. Lin and W. Zuo. Multi-level wavelet-CNN for image restoration. *CVPR Workshops*, 2018.
13. J. Long, E. Shelhamer and T. Darrell. Fully convolutional networks for semantic segmentation. *CVPR*, 2015.
14. Z. Zhang, Q. Liu and Y. Wang. Road extraction by deep residual U-Net. *IEEE GRSL* 15(5), 2018.
15. Y. Wang, J. Seo and T. Jeon. NL-LinkNet: Toward lighter but more accurate road extraction with non-local operations. arXiv:1908.08223, 2019.
16. G. Máttyus, W. Luo and R. Urtasun. DeepRoadMapper: Extracting road topology from aerial images. *ICCV*, 2017.
17. A. Mosinska, P. Márquez-Neila, M. Koziński and P. Fua. Beyond the pixel-wise loss for topology-aware delineation. *CVPR*, 2018.
18. A. Batra et al. Improved road connectivity by joint learning of orientation and segmentation. *CVPR*, 2019.
19. J. Mei, R.-J. Li, W. Gao and M.-M. Cheng. CoANet: Connectivity attention network for road extraction from satellite imagery. *IEEE TIP* 30, 2021.
20. F. Bastani et al. RoadTracer: Automatic extraction of road networks from aerial images. *CVPR*, 2018.
21. S. He et al. Sat2Graph: Road graph extraction through graph-tensor encoding. *ECCV*, 2020.
22. C. Hetang et al. Segment anything model for road network graph extraction. *CVPR Workshops*, 2024.
23. E. Xie et al. SegFormer: Simple and efficient design for semantic segmentation with transformers. *NeurIPS*, 2021.
24. J. Tao et al. Seg-Road: A segmentation network for road extraction based on transformer and CNN with connectivity structures. *Remote Sensing* 15(6):1602, 2023.
25. URoadNet: Dual sparse attentive U-Net for multiscale road network extraction. arXiv:2412.17573, 2024.
26. Z. Zhang et al. DSFC-Net: A dual-encoder spatial and frequency co-awareness network for rural road extraction. arXiv:2602.01278, 2026.
27. T. Williams and R. Li. Wavelet pooling for convolutional neural networks. *ICLR*, 2018.
28. Q. Li, L. Shen, S. Guo and Z. Lai. Wavelet integrated CNNs for noise-robust image classification. *CVPR*, 2020.
29. Q. Li and L. Shen. WaveSNet: Wavelet integrated deep networks for image segmentation. arXiv:2005.14461, 2020.
30. G. Xu et al. Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation. *Pattern Recognition* 143, 2023.
31. J. Zong et al. PWFNet: frequency-aware road extraction with pyramidal wavelet convolution. *Remote Sensing* 17(16):2895, 2025.
