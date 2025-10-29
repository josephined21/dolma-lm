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

### Arabic
Overall Accuracy: 10.71% on 588 items

#### Top 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Engineering | 20.83 |
| Chemistry | 16.07 |
| Biology | 13.89 |
| Business | 12.50 |
| Law | 12.50 |

#### Bottom 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Health | 2.86 |
| Computer Science | 5.00 |
| Physics | 6.15 |
| Economics | 7.14 |
| Math | 7.35 |


### French
Overall Accuracy: 8.33% on 588 items

#### Top 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Philosophy | 16.00 |
| Chemistry | 14.29 |
| Engineering | 12.50 |
| Biology | 11.11 |
| Other | 10.87 |

#### Bottom 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| History | 0.00 |
| Health | 0.00 |
| Business | 2.50 |
| Physics | 4.62 |
| Economics | 7.14 |

### Chinese
Overall Accuracy: 11.39% on 588 items

#### Top 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Business | 25.00 |
| Economics | 21.43 |
| Engineering | 16.67 |
| Psychology | 15.00 |
| Law | 14.58 |

#### Bottom 5 Subjects
| Subject | Accuracy (%) |
|----------|---------------|
| Computer Science | 0.00 |
| Biology | 2.78 |
| Other | 4.35 |
| Physics | 6.15 |
| Health | 8.57 |