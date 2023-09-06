# RADE
code for paper 'Reference-Assisted Dialogue Evaluation for Open-Domain Dialogue (ACL2023)'.

# Introduction
Evaluating open-domain dialogue systems is challenging for reasons such as the one-to-many problem, i.e., many appropriate responses other than just the golden response.   
As of now, automatic evaluation methods need better consistency with humans, while reliable human evaluation can be time- and cost-intensive. 
To this end, we propose the Reference-Assisted  Dialogue Evaluation (RADE) approach under the multi-task learning framework, which leverages the pre-created utterance as reference other than the gold response to relief the one-to-many problem. 
Specifically, RADE explicitly compares reference and the candidate response to predict their overall scores.
Moreover, an auxiliary response generation task enhances prediction via a shared encoder.

# Dataset
To support RADE, we extend three datasets with additional rated responses other than just a golden response by human annotation.
We will release our datasets as soon as possible to support the future work.


# Todo
- [] Add the code to implement our result.
- [] The three new-collected datasets is on the way.


# Contact us
Mail to shizhl@mail.sdu.edu.cn



 
