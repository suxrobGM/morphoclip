<!--
MELBA cover letter, adapted from https://github.com/melba-journal/submission#cover-letter.
Before submitting: fill in the date and arXiv ID, confirm the AI-use and conflict statements
with all co-authors, then paste the text into Scholastica or attach it as a PDF.
-->

[DATE]

Dear Editor,

We hereby submit our paper entitled:

**MorphoCLIP: Text-Supervised Contrastive Learning for Perturbation Matching in Cell Painting Images**

By the following authors:

Sukhrobbek Ilyosbekov, Shubham Gajjar, Rongfei Jin (Northeastern University, Portland, ME, USA)

The paper presents a contrastive model that links Cell Painting microscopy profiles with text
descriptions of compound, CRISPR and ORF perturbations. It keeps frozen vision and language
backbones, trains on a single consumer GPU, and evaluates bidirectional retrieval on held-out
CPJUMP1 perturbations together with the standard CPJUMP1 benchmark. We report negative results
alongside positive ones, in particular that gene-compound matching remains unsolved, because we
believe the MELBA readership benefits from an honest account of where text supervision helps
biological imaging and where it does not.

We acknowledge that:

- All authors have approved the submission for publication and consent to its review by MELBA.
- Authorship implies taking full responsibility for the accuracy and originality of the content
  of the submission.
- The work presented followed all appropriate ethical standards in conducting research and
  writing the manuscript, following all applicable laws and regulations regarding treatment of
  animals or human subjects.
- The code, configurations, split manifests and evaluation harness needed to replicate the results
  are publicly available at: <https://github.com/suxrobGM/morphoclip>

This is an original submission and there is no other paper (published or under review) with
significant overlapping content. A preprint of this manuscript is available on arXiv
([arXiv ID]); it has not been peer reviewed.

Use of generative AI: AI writing assistants were used to edit the manuscript for language and
clarity. They were not used for unsupervised content creation. All methods, experiments, analyses
and conclusions are the authors' own, and the authors reviewed every change.

Conflicts of interest: we declare no financial, commercial or personal conflicts of interest that
might have biased this work.

Review conflicts: we have no conflicts or concerns regarding the editors or potential reviewers.

Sincerely,

Sukhrobbek Ilyosbekov
Corresponding author, <ilyosbekov.s@northeastern.edu>
