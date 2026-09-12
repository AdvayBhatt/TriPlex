"""Fit and serialize only the selected model, independently of the experiment's ridge path."""
import numpy as np
from sklearn.linear_model import Ridge
from model_experiments import design, training_arrays
from triplex import ID, family


def predict_state(state, x, parent_x=None):
    def transform(prefix, values):
        values = values[:, state[prefix+'_keep']].astype(state[prefix+'_mean'].dtype)
        return (np.where(np.isnan(values), state[prefix+'_mean'], values)-state[prefix+'_mean'])/state[prefix+'_scale']
    values = transform('snp', x) @ state['snp_coef'].T + state['snp_intercept']
    if str(state['kind']) == 'flat':
        return values[:, 0]
    if parent_x is None:
        raise ValueError('Parent inputs required')
    valid = np.isfinite(parent_x).any(axis=1)
    fp = transform('parent', parent_x) @ state['parent_coef'] + state['parent_intercept']
    return np.where(valid, fp + values[:, 1], values[:, 0])


def fit_selected(history, metadata, geno, parents, model):
    if model not in ['parents_10000', 'flat_3000000']:
        raise ValueError('Unsupported fixed model policy')
    level, raw, candidates, target = training_arrays(history, metadata, geno)
    x, _, prep = design(raw, target, .5, .01)
    state = {'kind': np.array('parents' if model.startswith('parents') else 'flat'), 'markers': geno.markers}
    for key in ['keep', 'mean', 'scale']:
        state['snp_'+key] = prep[key]
    family_labels = level[ID].map(family)
    family_mean = level.groupby(family_labels).y.mean()
    if model.startswith('parents'):
        y = np.column_stack([level.y, level.y-family_labels.map(family_mean)])
        fit = Ridge(alpha=30000., solver='cholesky').fit(x, y)
    else:
        fit = Ridge(alpha=3000000., solver='cholesky').fit(x, level.y.to_numpy())
    state['snp_coef'], state['snp_intercept'] = np.atleast_2d(fit.coef_), np.atleast_1d(fit.intercept_)
    parent_x = parents.reindex(candidates.population).to_numpy()
    if model.startswith('parents'):
        families = family_mean.index.intersection(parents.index)
        px, _, prep = design(parents.reindex(families).to_numpy(), parent_x, 1., 0.)
        parent_fit = Ridge(alpha=10000., solver='cholesky').fit(px, family_mean.reindex(families).to_numpy())
        for key in ['keep', 'mean', 'scale']:
            state['parent_'+key] = prep[key]
        state['parent_coef'], state['parent_intercept'] = parent_fit.coef_, parent_fit.intercept_
    prediction = predict_state(state, target, parent_x)
    return candidates, prediction, state
