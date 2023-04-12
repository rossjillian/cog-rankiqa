# Cog Implementation of RankIQA

This is an implementation of [RankIQA](https://github.com/YunanZhu/Pytorch-TestRankIQA) as a Cog model. [Cog](https://github.com/replicate/cog) packages machine learning models as standard containers.

First, download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1OQ0IQrWoricMhaIyfwqsJVlYpXHKPP1z). Note that these weights have been provided by [this GitHub repository](https://github.com/YunanZhu/Pytorch-TestRankIQA). Download at your own risk.

Then, you can run predictions:

`cog predict -i image=@test.png`
