### Answer to the approved question

On this held-out group split, **MiniLM detects misspelled spam more reliably by recall**.
For the mixed condition it detects 95 of 109 spam messages (87.16%), compared with
89 of 109 (81.65%) for TF-IDF: an advantage of 5.50 percentage points, or six messages.
Its full-set mixed accuracy is 97.70% and macro F1 is 0.9501, versus 97.31% and 0.9395.
It also has higher recall in deletion-only and transposition-only conditions.

The tradeoff is false positives: MiniLM flags 4 of 672 ham messages in the mixed
condition (0.60%), versus 1 for TF-IDF (0.15%). MiniLM therefore is not superior on every
practical criterion. From clean to mixed, TF-IDF recall decreases by 1.83 points and
MiniLM recall increases by 1.83 points. Small perturbations can move borderline examples
in either direction; the improvement does not mean misspellings inherently help a model.

The mixed paired recall-difference interval is 0.00 to 11.01 points and exploratory
exact McNemar p=0.1094. With only 109 spam examples, these results do not establish
population-wide superiority. Message-level bootstrap intervals may understate uncertainty
from related messages. Report the observed advantage and uncertainty together.

### Scope and limitations

One older English corpus, one group split, one typo seed and one internal edit per message
limit external validity. Only 707/781 mixed messages are modified; all 109 spam messages
are modified. The full-set analysis includes 74 unchanged ham messages, while modified-only
results compare the same eligible cohort before and after editing. The safety lexicon reduces
real-word substitutions but does not guarantee unchanged semantics. Numbers and URLs are protected.
Class proportions differ somewhat between group partitions; the development set contains only
71 spam messages. Neither model uses sender information, and output scores are uncalibrated.
The encoder may truncate unusually long inputs at 256 wordpieces. Grouping reduces related-text
leakage but cannot identify every campaign relationship or rule out overlap in unknown external
pretraining data. Inspect raw messages carefully before treating annotation ambiguity as an error.

A useful next experiment is character n-gram features or training-only typo augmentation,
evaluated using a newly held-out group split and multiple corruption seeds. Do not tune this
improvement on the reported test set. In practice, these classifiers should assist review rather
than automatically delete messages based on an unvalidated threshold.
