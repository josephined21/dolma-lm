## Benchmarking Results
The benchmarking script evaluates model accuracy on MMLU-ProX-Lite for a given language. It loads the `test` split (falling back to `validation`), formats each multiple-choice question with options A–J, and wraps it in a short system/user chat prompt that instructs the model to answer with a single letter. Instead of free-form generation, it uses a likelihood-based multiple-choice method: for each question it appends each candidate letter to the prompt, masks prompt tokens, computes the average token-level negative log-likelihood over just the candidate suffix, and chooses the option with the lowest loss. It then reports overall accuracy and per-subject accuracy from the dataset’s `category` field.

### English
Overall Accuracy: 32.14% on 588 items

#### Top 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Biology | 63.89 |
| Psychology | 60.00 |
| Economics | 47.62 |
| Health | 42.86 |
| History | 42.11 |

#### Bottom 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Computer Science | 15.00 |
| Chemistry | 16.07 |
| Law | 18.75 |
| Business | 22.50 |
| Other | 23.91 |

### Spanish
Overall Accuracy: 10.03% on 588 items

#### Top 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Math | 14.71 |
| Engineering | 14.58 |
| Biology | 13.89 |
| Law | 12.00 |
| Philosophy | 12.00 |

#### Bottom 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Physics | 3.08 |
| Computer Science | 5.00 |
| Health | 5.71 |
| Other | 8.70 |
| Chemistry | 8.93 |
