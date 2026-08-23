# 🧠 NeuroCanvas: 6D Directed Coherence Wavefield & 3D Tonal Torus Neurofeedback Engine

**NeuroCanvas** is a real-time, closed-loop Brain-Computer Interface (BCI) and neurofeedback platform that translates multi-channel electroencephalographic (EEG) phase dynamics into an immersive musical state-space. 

By integrating **6D directed phase-locking graphs (iPLV)** across 16 micro-electrodes with the **topological Tonal Torus ($\mathbb{T}^2$) of working memory**, the system allows the human brain to perceive, navigate, and steer its own spatiotemporal cognitive trajectories through auditory predictive coding.

## 🎥 Real-Time Neurofeedback Demos

### 1. Dark Psybient & Continuous Torus Navigation
[![NeuroCanvas Psybient Demo](https://img.youtube.com/vi/P_fH1Smlr6Y/maxresdefault.jpg)](https://www.youtube.com/watch?v=P_fH1Smlr6Y "Watch NeuroCanvas Demo 1")

### 2. Melodic Full-On Psytrance & Modal Percussion
[![NeuroCanvas Full-On Demo](https://img.youtube.com/vi/kbu9Etb1WEo/maxresdefault.jpg)](https://www.youtube.com/watch?v=kbu9Etb1WEo "Watch NeuroCanvas Demo 2")

---

## 📑 Table of Contents
1. [Theoretical & Neuroscientific Foundations](#1-theoretical--neuroscientific-foundations)
   - [1.1 Prefrontal Tonal Manifold ($\mathbb{T}^2$)](#11-prefrontal-tonal-manifold-mathbft2)
   - [1.2 Working Memory 2.0: Theta-Gamma Phase Multiplexing & Temporal Coding](#12-working-memory-20-theta-gamma-phase-multiplexing--temporal-coding)
   - [1.3 6D Directed Phase-Lag Connectivity (iPLV)](#13-6d-directed-phase-lag-connectivity-iplv)
   - [1.4 Auditory Predictive Coding & Sensorimotor Closed-Loop](#14-auditory-predictive-coding--sensorimotor-closed-loop)
2. [Mathematical Formulations & 4-Axis Decomposition](#2-mathematical-formulations--4-axis-decomposition)
   - [2.1 4-Axis Decomposition of the Intention Vector](#21-4-axis-decomposition-of-the-intention-vector)
   - [2.2 120-Edge Physical Topology & Modal Percussion Matrix](#22-120-edge-physical-topology--modal-percussion-matrix)
   - [2.3 Weber-Fechner Logarithmic Companding](#23-weber-fechner-logarithmic-companding)
   - [2.4 Temporal Dynamics & Rhythm Engine ($ry$ & $\mathbf{S}_{32}$)](#24-temporal-dynamics--rhythm-engine-ry--mathbfs_32)
3. [Audio Engine Architecture](#3-audio-engine-architecture)
   - [3.1 12-Channel Persistent Phase Banks & Sample-Accurate Linear Ramping](#31-12-channel-persistent-phase-banks--sample-accurate-linear-ramping)
   - [3.2 Continuous Morphing: Dark Psybient $\longleftrightarrow$ Full-On Psytrance](#32-continuous-morphing-dark-psybient-longleftrightarrow-full-on-psytrance)
4. [Scientific References & DOIs](#4-scientific-references--dois)
5. [Quickstart & Execution](#5-quickstart--execution)

---

## 🧬 1. Theoretical & Neuroscientific Foundations

```
   ┌─────────────────────────────────────────────────────────────┐
   │                  HUMAN AUDITORY CORTEX                      │
   │  Auditory Predictive Coding & Tonal Expectation (Vuust 2022) │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Evokes Neural State Vector
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             PREFRONTAL WORKING MEMORY DYNAMICS              │
   │   Theta-Gamma Phase Multiplexing (Lisman 2013; Miller 2018) │
   │   - Low Gamma (30–50 Hz)  --> Retrospective Past (S₀)       │
   │   - High Gamma (60–85 Hz) --> Prospective Future (S₃₁)      │
   │   - Sagitta κ(u)          --> Harmonic Tension & Acid Sweep │
   │   - Temporality ry        --> Rolling Bass Momentum & Drive │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ 26mm Sensor Scalp Potential
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │           16-ELECTRODE / 120-EDGE iPLV TENSOR               │
   │  Volume-Conduction-Free Directed Phase Graph (Bruña 2018)   │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Real-Time GPU Tensor Projection
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │               3D TONAL TORUS GEODESIC ENGINE                │
   │     Janata Tonal Manifold T² = S¹ × S¹ (Janata 2002)        │
   │     - θ: Circle of Fifths (+7 semitones)                    │
   │     - φ: Circle of Major Thirds (+4 semitones)              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Continuous Multi-Voice Synthesis
                                  ▼
                        CLOSED-LOOP AUDIO STREAM
```

### 1.1 Prefrontal Tonal Manifold ($\mathbb{T}^2$)
Functional neuroimaging demonstrates that the human brain organizes musical harmony, pitch relationships, and tonal expectations along a continuous, boundaryless **two-dimensional torus ($\mathbb{T}^2 = S^1 \times S^1$)** located in the rostromedial prefrontal cortex (rmPFC) [Janata et al., 2002] [1]:
* **Major Dimension ($\theta$):** The Circle of Fifths ($+7$ semitones), wrapping periodically every 12 chromatic steps.
* **Minor Dimension ($\phi$):** The Circle of Major Thirds ($+4$ semitones), wrapping periodically every 3 steps ($C \to E \to G\sharp \to C$).

In NeuroCanvas, navigation is unconstrained: the cognitive state vector glides along geodesic trajectories on the toroidal surface without arbitrary Cartesian boundaries.

### 1.2 Working Memory 2.0: Theta-Gamma Phase Multiplexing & Temporal Coding
Under the **Working Memory 2.0** framework [Miller, Lundqvist, & Bastos, 2018] [2], cognitive representations are time-multiplexed into discrete gamma-band bursts nested within slower theta oscillations (4–8 Hz) [Lisman & Jensen, 2013] [5]:
* **Low Gamma (30–50 Hz, Early Theta Phase):** Anchors **Retrospective Retrieval ($S_0$, Past)** [Colgin et al., 2009] [7].
* **High Gamma (60–85 Hz, Late Theta Phase):** Encodes **Prospective Prediction ($S_{31}$, Future)** [Heusser et al., 2016] [8].
* **Temporality ($ry$):** Represents the dynamic ratio of phase velocity between retrospective grounding and prospective momentum. When prospective drive dominates, the temporal axis accelerates the rhythmic engine into an offbeat rolling gallop.

### 1.3 6D Directed Phase-Lag Connectivity (iPLV)
To eliminate volume conduction and source leakage across the 26mm 16-channel array, NeuroCanvas implements the reformulated **imaginary Phase Locking Value (iPLV)** [Bruña, Maestú, & Pereda, 2018] [3]:
$$\text{iPLV}_{i,j} = \frac{1}{T} \Im \left\{ \sum_{t=1}^T \dot{x}_i(t) \cdot \dot{x}_j^*(t) \right\} = \frac{1}{T} \sum_{t=1}^T \sin(\varphi_i(t) - \varphi_j(t))$$
The sign of $\text{iPLV}_{i,j}$ determines the physical vector of traveling phase waves across the cortical patch ($\text{sign} > 0 \implies i \to j$; $\text{sign} < 0 \implies j \to i$).

### 1.4 Auditory Predictive Coding & Sensorimotor Closed-Loop
The brain processes auditory input via hierarchical predictive coding [Vuust et al., 2022; Koelsch, 2014] [4, 6]. NeuroCanvas establishes a closed sensory-motor feedback loop:
1. The engine synthesizes a polyphonic harmonic state on the Tonal Torus.
2. The auditory-prefrontal cortex anticipates the resolution, generating an intentional trajectory ($\text{iPLV}_{32 \times 120}$).
3. The engine decodes the trajectory in real time, modulates the tonal center, and returns audible confirmation to the auditory cortex.

---

## 📐 2. Mathematical Formulations & 4-Axis Decomposition

### 2.1 4-Axis Decomposition of the Intention Vector
From the 32-point phase-space trajectory $\mathbf{S} = \{s_0, s_1, \dots, s_{31}\}$, four canonical control axes are extracted:

| Axis | Metric Name | Mathematical Definition | Acoustic Function in Synthesis |
| :--- | :--- | :--- | :--- |
| **$lx$** | **Tonal Longitude** | $\Delta x = \frac{s_{31}^x - s_0^x}{\|\mathbf{d}\|}$ | Navigation along Circle of Fifths ($\theta$) [Janata 2002] [1] |
| **$ly$** | **Tonal Latitude** | $\Delta y = \frac{s_{31}^y - s_0^y}{\|\mathbf{d}\|}$ | Navigation along Major/Minor Thirds ($\phi$) [Janata 2002] [1] |
| **$rx$** | **Sagitta $\kappa(u)$** | $\frac{1}{16\|\mathbf{d}\|} \sum_{k=1}^{30} \left( \Delta x \cdot py_k - \Delta y \cdot px_k \right)$ | Lateral trajectory curvature $\to$ TB-303 Acid Resonance / Squelch [Koelsch 2014] [6] |
| **$ry$** | **Temporal Bias** | $\frac{\|\mathbf{S}_{future}\| - \|\mathbf{S}_{past}\|}{\|\mathbf{S}_{future}\| + \|\mathbf{S}_{past}\| + \epsilon}$ | Phase acceleration $\to$ K-B-B-B Rolling Bass Drive & Momentum [Colgin 2009; Heusser 2016] [7, 8] |

### 2.2 120-Edge Physical Topology & Modal Percussion Matrix
The 120 electrode pairs $(i, j)$ on the 26mm sensor are partitioned into **3 physical percussion registers** based on their spatial euclidean length $L_p = \sqrt{\Delta X_p^2 + \Delta Y_p^2}$:
1. **Short Links ($L_p < 8\text{ mm}$):** High-frequency micro-shakers and glitch clicks ($5\text{–}10\text{ kHz}$).
2. **Medium Links ($8 \le L_p < 15\text{ mm}$):** Tonal FM Bongos, tablas, and acid zaps quantized to Torus degrees.
3. **Long Links ($L_p \ge 15\text{ mm}$):** Snare, clap, and noise splash transients.

### 2.3 Weber-Fechner Logarithmic Companding
To prevent acoustic masking while retaining 100% of all 120 active connections without muting:
$$\text{Amp}_p = \text{SoftFloor} + (1 - \text{SoftFloor}) \cdot \frac{\ln(1 + 4 \cdot \text{NormPower}_p)}{\ln(5)}$$
All 120 edges remain audible ($\ge 20\%$ presence), with dominant functional networks rising to the foreground while subtle sub-networks provide a rich, transparent spatial bed.

### 2.4 Temporal Dynamics & Rhythm Engine ($ry$ & $\mathbf{S}_{32}$)
In traditional electronic music, rhythm is generated via rigid, hardcoded clock divisions (e.g., static 140 BPM grid), which ignores the brain's real-time temporal mechanics. In **NeuroCanvas**, the entire rhythm and drum architecture is **100% emergent from the 32 temporal slices of the working memory trajectory ($\mathbf{S}_{32} = \{s_0, s_1, \dots, s_{31}\}$) and the Temporal Bias axis ($ry$)**:

```
        ┌─────────────────── 32 TEMPORAL SLICES (traj_32) ───────────────────┐
        │  S₀ (0/4: KICK)  ──►  S₈ (1/4: BASS 1)  ──►  S₁₆ (2/4: BASS 2)  ──►  S₂₄ (3/4: BASS 3)  ──►  S₃₁ (FUTURE) │
        └────────────────────────────────────┬────────────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            DYNAMIC ACCENTS & SWING                     ROLLING BASS MOMENTUM (ry)
      Segment Velocities ||Sₖ₊₁ - Sₖ||           ry < 0: Heavy Sub-Body / Kick Focus
      govern per-hit velocity & dynamics         ry > 0: High-Energy 16th Gallop Drive
```

#### 1. 4-Segment Trajectory Partitioning for the K-B-B-B Grid
Instead of static mechanical triggers, the 4 subdivisions of the quarter-beat measure are bound directly to the 4 cardinal checkpoints of the 32-slice trajectory:
* **$S_0 \to S_8$ (Downbeat Segment / Past):** Drives the **Kick Punch & Body Energy ($w_{kick}$)**.
* **$S_8 \to S_{16}$ (First Offbeat Segment):** Drives the velocity of **Bass Hit 1 ($w_{b1}$)**.
* **$S_{16} \to S_{24}$ (Midpoint Bridge Segment):** Drives the velocity of **Bass Hit 2 & Open Hat ($w_{b2}$)**.
* **$S_{24} \to S_{31}$ (Leading Future Segment):** Drives the velocity of **Bass Hit 3 ($w_{b3}$)**.

The dynamic velocity weight of each rhythmic step is computed from the real-time physical step distance in the brain's phase space:
$$v_0 = \|s_8 - s_0\|, \quad v_1 = \|s_{16} - s_8\|, \quad v_2 = \|s_{24} - s_{16}\|, \quad v_3 = \|s_{31} - s_{24}\|$$
$$w_{step} = \text{clamp}\left(1.8 \cdot \frac{v_k}{\sum v_i}, 0.5, 1.4\right)$$

#### 2. Temporal Bias ($ry$) as Rolling Bass Momentum
The global Temporal Bias $ry$ determines the balance of momentum between retrospective anchor and prospective drive:
$$ry = \frac{\sum_{k=16}^{31} \text{HighGamma}_k - \sum_{k=0}^{15} \text{LowGamma}_k}{\sum_{k=16}^{31} \text{HighGamma}_k + \sum_{k=0}^{15} \text{LowGamma}_k + \epsilon}$$

* **Retrospective Focus ($ry < 0$ / Low Gamma Dominance):**
  The rhythmic energy settles into the foundational **Kick and Sub-Body ($S_0$)**. The offbeat bass softens into a wide, sustained legato drone.
* **Prospective Focus ($ry > 0$ / High Gamma Anticipation):**
  The rhythmic energy surges into the **Offbeat Rolling Bass (the 3 "B"s)**. The bass envelope decay tightens ($\text{decay} \propto 6.0 + 3.0 \cdot ry$), creating an aggressive, driving 16th-note gallop that physically propels the musical progression forward.

#### 3. Dynamic Tempo Scaling ($135 \text{--} 146\text{ BPM}$)
The instantaneous tempo of the track breathes continuously with the aggregate trajectory velocity:
$$\text{Live BPM} = 136.0 + \text{clamp}\left(2.5 \cdot \sum_{i=0}^3 v_i, 0.0, 10.0\right)$$
This anchors the groove strictly within the natural human psytrance perception window ($135\text{--}146\text{ BPM}$, corresponding to an optimal $105\text{ ms}$ per 16th-note), preventing auditory roughness/rattling while maintaining 100% biological synchronization.

---

## 🏗️ 3. Audio Engine Architecture

### 3.1 12-Channel Persistent Phase Banks & Sample-Accurate Linear Ramping
To eradicate audio clicks, pops, frequency sweeps (chirps), and buffer underruns:
1. **12 Persistent Phase Banks:** All 12 key centers maintain continuous phase accumulators on CUDA. Frequencies are static and never jump.
2. **Sample-Accurate Linear Ramping (`torch.linspace`):** Amplitudes are interpolated linearly across every individual sample in a block, completely eliminating zipper noise.
3. **Equal-Power Cosine Crossfading:**
   $$G_{low} = \cos\left(\text{blend} \cdot \frac{\pi}{2}\right), \quad G_{high} = \sin\left(\text{blend} \cdot \frac{\pi}{2}\right)$$

### 3.2 Continuous Morphing: Dark Psybient $\longleftrightarrow$ Full-On Psytrance
The real-time parameter $\text{Style Morph} \in [0.0, 1.0]$ smoothly transforms the acoustic landscape:
* **$\text{Morph} = 0.0$ (Dark Psybient):** Continuous holographic cosmic organ pad with live Theta-frequency phase breathing.
* **$\text{Morph} = 1.0$ (Full-On Psytrance):** 16th-note K-B-B-B Rolling Bass, punchy kick ($220\text{ Hz} \to 48\text{ Hz}$), sidechain compression, and galloping 16th-note Acid 303 arpeggiators.

---

## 📚 4. Scientific References & DOIs

1. **Janata, P., Birk, J. L., Van Horn, J. D., Leman, M., Tillmann, B., & Bharucha, J. J. (2002).**  
   *The Cortical Topography of Tonal Structures Underlying Western Music.*  
   **Science**, 298(5601), 2167–2170.  
   DOI: [10.1126/science.1076262](https://doi.org/10.1126/science.1076262)

2. **Miller, E. K., Lundqvist, M., & Bastos, A. M. (2018).**  
   *Working Memory 2.0.*  
   **Neuron**, 100(2), 463–475.  
   DOI: [10.1016/j.neuron.2018.09.023](https://doi.org/10.1016/j.neuron.2018.09.023)

3. **Bruña, R., Maestú, F., & Pereda, E. (2018).**  
   *Phase Locking Value revisited: teaching new tricks to an old dog.*  
   **Journal of Neural Engineering**, 15(5), 056011.  
   DOI: [10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)

4. **Vuust, P., Heggli, O. A., Friston, K. J., & Kringelbach, M. L. (2022).**  
   *Music in the brain: From perception to expectation and pleasure.*  
   **Nature Reviews Neuroscience**, 23(5), 287–305.  
   DOI: [10.1038/s41583-022-00578-5](https://doi.org/10.1038/s41583-022-00578-5)

5. **Lisman, J. E., & Jensen, O. (2013).**  
   *The Theta-Gamma Neural Code.*  
   **Neuron**, 77(6), 1002–1016.  
   DOI: [10.1016/j.neuron.2013.03.007](https://doi.org/10.1016/j.neuron.2013.03.007)

6. **Koelsch, S. (2014).**  
   *Brain correlates of music-evoked emotions.*  
   **Nature Reviews Neuroscience**, 15(3), 170–180.  
   DOI: [10.1038/nrn3666](https://doi.org/10.1038/nrn3666)

7. **Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., Moser, M. B., & Moser, E. I. (2009).**  
   *Frequency of gamma oscillations routes flow of information in the hippocampus.*  
   **Nature**, 462(7271), 353–357.  
   DOI: [10.1038/nature08447](https://doi.org/10.1038/nature08447)

8. **Heusser, A. C., Poeppel, D., Ezzyat, Y., & Davachi, L. (2016).**  
   *Episodic sequence memory is supported by a theta–gamma phase code.*  
   **Nature Neuroscience**, 19(10), 1374–1380.  
   DOI: [10.1038/nn.4374](https://doi.org/10.1038/nn.4374)

9. **Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., Moser, M. B., & Moser, E. I. (2022).**  
   *Toroidal topology of population activity in grid cells.*  
   **Nature**, 602(7895), 123–128.  
   DOI: [10.1038/s41586-021-04268-7](https://doi.org/10.1038/s41586-021-04268-7)

10. **Lundqvist, M., Rose, J., Herman, P., Brincat, S. L., Buschman, T. J., & Miller, E. K. (2016).**  
    *Gamma and Beta Bursts Underlie Working Memory.*  
    **Neuron**, 90(1), 152–164.  
    DOI: [10.1016/j.neuron.2016.02.028](https://doi.org/10.1016/j.neuron.2016.02.028)

11. **Bastos, A. M., Loonis, R., Kornblith, S., Lundqvist, M., & Miller, E. K. (2018).**  
    *Laminar recordings in frontal cortex suggest distinct layers for maintenance and control of working memory.*  
    **Proceedings of the National Academy of Sciences (PNAS)**, 115(5), 1117–1122.  
    DOI: [10.1073/pnas.1710323115](https://doi.org/10.1073/pnas.1710323115)

12. **Albouy, P., Weiss, A., Baillet, S., & Zatorre, R. J. (2017).**  
    *Selective Theta-Band Stimulation of the Frontal-Parietal Network Enhances Auditory Working Memory.*  
    **Neuron**, 94(1), 193–206.  
    DOI: [10.1016/j.neuron.2017.02.044](https://doi.org/10.1016/j.neuron.2017.02.044)

13. **Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).**  
    *Identifying true brain interaction from EEG data using the imaginary part of coherency.*  
    **Clinical Neurophysiology**, 115(10), 2292–2307.  
    DOI: [10.1016/j.clinph.2004.04.029](https://doi.org/10.1016/j.clinph.2004.04.029)

14. **Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).**  
    *Phase lag index: Assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources.*  
    **Human Brain Mapping**, 28(11), 1178–1193.  
    DOI: [10.1002/hbm.20346](https://doi.org/10.1002/hbm.20346)

15. **Lerdahl, F. (2001).**  
    *Tonal Pitch Space.*  
    **Oxford University Press**, New York.  
    ISBN: `9780195178296`

---

## ⚡ 5. Quickstart & Execution

```bash
# 1. Install dependencies
pip install numpy pygame sounddevice torch pylsl

# 2. Run the Closed-Loop GPU Torus Synthesizer
python3 neuro_janata_musical_core.py
```

* **Dancefloor Style Morph:** `W` / `UP` — morph toward **Full-On Psytrance**, `S` / `DOWN` — morph toward **Dark Psy-Chill**.
* **Navigation:** The musical state automatically follows the real-time intention vector $[lx, ly]$ streamed from the central CUDA daemon in `neuro_heterarchy_core.py`.
