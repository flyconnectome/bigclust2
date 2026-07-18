# Find what separates two groups

The **Feature Comparison** widget takes two sets of neurons and ranks the
features that distinguish them. It answers "these two blobs look different —
what is actually different about them?", and its companion dialog answers "should
these be one group or two?".

It needs a `features` matrix. With only distances there is nothing to rank.

## Steps

1. Select the first group of neurons in the scatter plot.
2. **View → Feature Comparison** (++shift+cmd+f++).
3. In the **Groups** box, under **Group A**, press **Copy Selection**.
4. Go back to the scatter plot and select the second group.
5. Under **Group B**, press **Copy Selection**.
6. Press **Refresh**.

The **Table** tab now lists features ranked by how well they separate A from B.
The **Graph** tab plots the top-ranked ones as distributions, which is usually
the faster read — a feature with two cleanly separated humps is a real
difference; one with two overlapping bell curves is not, whatever its score says.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `feature-comparison.gif`<br>
Select cluster A, Copy Selection; select cluster B, Copy Selection; Refresh; the ranked table fills.
  </div>
  <figcaption>Ranking the features that separate two clusters.</figcaption>
</figure>

**Edit** next to either group opens a dialog where you can pick members by ID or
label, filter them, and pull in the current figure selection — use it when the
group you want is not something you can lasso.

## Preparing the features

The **Data** box controls what is being compared, and getting this right matters
more than the choice of scoring metric.

| Control | What it does |
|---|---|
| `Aggregation` | How feature values are combined per group |
| `Top-level filters` | Restrict to some feature groups (upstream vs downstream, say). Only present if your `features` file has MultiIndex columns. |
| `Min synapse count` | Drop weak connections before ranking. Raise it to stop the ranking being dominated by single-synapse noise. |
| `Normalize per neuron` | Compare connection *fractions* rather than raw counts, so a large neuron with many synapses is not automatically different from a small one. |

For connectivity features, `Normalize per neuron` on and a `Min synapse count` of
at least 2 or 3 is a sane starting point.

## Scoring

The **Scoring** box picks how "separates the groups" is measured. Two families:

**L1 (sparse logistic regression)** fits a classifier that is penalised for using
many features, so what survives is a *small set* of features that jointly
distinguish the groups.

| Control | Notes |
|---|---|
| `L1 C` | Inverse regularisation strength. Lower keeps fewer features. |
| `Solver` | `liblinear` or `saga` |
| `Max iter` | Raise if the fit warns about convergence |
| `Standardize features` | Put features on a common scale first — usually yes |
| `Use stability selection` | Refit over bootstrap resamples and keep what is consistently selected |
| `Bootstraps` / `Sample frac` | The resampling for the above |

Turn on stability selection when you intend to *report* the result. A single L1
fit will happily pick one of two correlated features arbitrarily; stability
selection tells you which choices were robust.

**Permutation importance** measures how much the score degrades when each feature
is shuffled, one at a time. Slower, but it does not assume a linear boundary.

| Control | Notes |
|---|---|
| `Repeats` | More is more stable and slower |
| `Scoring` | The metric being degraded |
| `Evaluation` | `In-sample`, `3-fold CV` or `5-fold CV` |

!!! warning "In-sample scores are optimistic"

    `In-sample` evaluation scores the model on the data it was fitted to. With
    wide feature matrices and small groups — which is the normal situation in
    connectomics — a model can separate any two arbitrary sets of neurons
    perfectly and tell you nothing.

    Use `5-fold CV` before believing a separation. If the cross-validated score
    collapses relative to the in-sample one, the "difference" you found was the
    model memorising your groups.

## Is this one cluster or two?

**Eval. separation** opens a dialog that scores the *split itself* rather than
ranking features. Pick a distance metric, optionally tick **Trim outliers**, and
you get a small table of `Metric` / `Score` / `Interpretation` rows.

This is the check to run before you split a cluster in the [Cluster
tab](recluster.md#fixing-it-by-hand) or push a new type name. Two groups that no
metric distinguishes and whose top-ranked feature has overlapping distributions
are one group you have drawn a line through.

## Reading a single feature

Click any row in the ranked table to open a detail dialog for that feature alone,
with its own distribution plot and a **Show KDE** toggle. This is where you check
whether a high-ranking feature is separating the groups the way you assume, or
whether the score is being carried by a handful of outliers.

## Getting it out

The table's export writes the ranking to `feature_ranking.csv`. **Always on top**
keeps the window above the main one, which is what you want while you go back and
forth adjusting the selection.

Next: [refresh the meta data](update-meta-data.md).
