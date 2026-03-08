"""
Curriculum Pseudo-Labeling for Learning Style Identification (CPL-LS)

Replaces sklearn's SelfTrainingClassifier with dynamic, per-class thresholds
inspired by FlexMatch (Zhang et al., NeurIPS 2021). Instead of a single fixed
confidence cutoff, thresholds adapt each iteration based on how well each class
has been learned so far.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class CurriculumSelfTraining(BaseEstimator, ClassifierMixin):
    """
    CPL-LS: Curriculum Pseudo-Labeling wrapper for any sklearn-compatible classifier.

    Parameters
    ----------
    base_estimator : estimator
        Any classifier exposing fit, predict, predict_proba (sklearn API).
    tau_ref : float, default=0.95
        Reference confidence threshold (corresponds to τ_ref in the paper).
    max_iter : int, default=5
        Maximum number of self-training iterations.
    epsilon : float, default=0.05
        Warm-up threshold: if the best-learned class has learning effect
        below epsilon, all thresholds are set to 0 (warm-up phase).
    verbose : bool, default=True
        Whether to print iteration-level diagnostics.
    """

    def __init__(self, base_estimator, tau_ref=0.95, max_iter=5,
                 epsilon=0.05, verbose=True):
        self.base_estimator = base_estimator
        self.tau_ref = tau_ref
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose

        # Will be populated during fit
        self.model_ = None
        self.classes_ = None
        self.threshold_history_ = []

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Run the CPL-LS self-training loop.

        Parameters
        ----------
        X_labeled : array-like of shape (n_labeled, n_features)
        y_labeled : array-like of shape (n_labeled,)
        X_unlabeled : array-like of shape (n_unlabeled, n_features)

        Returns
        -------
        self
        """
        # Convert to numpy for consistency
        X_l = np.array(X_labeled) if not isinstance(X_labeled, np.ndarray) else X_labeled.copy()
        y_l = np.array(y_labeled) if not isinstance(y_labeled, np.ndarray) else y_labeled.copy()
        X_u = np.array(X_unlabeled) if not isinstance(X_unlabeled, np.ndarray) else X_unlabeled.copy()

        self.classes_ = np.unique(y_l)
        self.threshold_history_ = []

        # Clone the base estimator so we don't mutate the original
        self.model_ = clone(self.base_estimator)

        for iteration in range(1, self.max_iter + 1):
            if len(X_u) == 0:
                if self.verbose:
                    print(f"      [CPL iter {iteration}] No unlabeled samples left. Stopping.")
                break

            # Step 1: Train on current labeled pool
            self.model_.fit(X_l, y_l)

            # Step 2: Predict probabilities on unlabeled pool
            proba = self.model_.predict_proba(X_u)
            predicted_classes = self.classes_[np.argmax(proba, axis=1)]
            max_proba = np.max(proba, axis=1)

            # Step 3: Compute per-class learning effect σ_t(c)
            sigma = {}
            for c in self.classes_:
                # Fraction of unlabeled samples confidently predicted as class c
                confident_mask = max_proba > self.tau_ref
                class_mask = predicted_classes == c
                sigma[c] = np.sum(confident_mask & class_mask) / len(X_u)

            max_sigma = max(sigma.values())

            # Step 4: Compute dynamic thresholds
            thresholds = {}
            if max_sigma < self.epsilon:
                # Warm-up phase: model hasn't learned enough yet, accept everything
                for c in self.classes_:
                    thresholds[c] = 0.0
                phase = "warm-up"
            else:
                # Normal CPL: threshold proportional to learning progress
                for c in self.classes_:
                    beta_c = sigma[c] / max_sigma if max_sigma > 0 else 0.0
                    thresholds[c] = beta_c * self.tau_ref
                phase = "curriculum"

            self.threshold_history_.append({
                'iteration': iteration,
                'phase': phase,
                'sigma': dict(sigma),
                'thresholds': dict(thresholds)
            })

            # Step 5: Select pseudo-labels that exceed their class-specific threshold
            selected_mask = np.zeros(len(X_u), dtype=bool)
            for i in range(len(X_u)):
                pred_class = predicted_classes[i]
                if max_proba[i] > thresholds[pred_class]:
                    selected_mask[i] = True

            n_selected = np.sum(selected_mask)

            if self.verbose:
                thresh_str = ", ".join(
                    f"c{c}={thresholds[c]:.3f}" for c in self.classes_
                )
                sigma_str = ", ".join(
                    f"c{c}={sigma[c]:.4f}" for c in self.classes_
                )
                print(f"      [CPL iter {iteration}] phase={phase} | "
                      f"σ=[{sigma_str}] | T=[{thresh_str}] | "
                      f"selected={n_selected}/{len(X_u)}")

            if n_selected == 0:
                if self.verbose:
                    print(f"      [CPL iter {iteration}] No candidates above threshold. Stopping.")
                break

            # Step 6: Promote selected pseudo-labels into the labeled pool
            X_selected = X_u[selected_mask]
            y_selected = predicted_classes[selected_mask]

            X_l = np.concatenate([X_l, X_selected], axis=0)
            y_l = np.concatenate([y_l, y_selected], axis=0)

            # Remove promoted samples from unlabeled pool
            X_u = X_u[~selected_mask]

        # Final training on the fully expanded labeled pool
        self.model_.fit(X_l, y_l)

        if self.verbose:
            print(f"      [CPL] Finished. Final labeled pool size: {len(X_l)} "
                  f"(started with {len(X_labeled)})")

        return self

    def predict(self, X):
        """Predict class labels for X."""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return self.model_.predict_proba(X)

    def score(self, X, y):
        """Return accuracy on test data."""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        preds = self.predict(X)
        return np.mean(preds == y)
