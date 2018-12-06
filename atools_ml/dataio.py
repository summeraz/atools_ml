import os

import numpy as np
import pandas as pd
import signac


def signac_to_df(project):
    df_index = pd.DataFrame(project.index())
    df_index = df_index.set_index(['_id'])
    statepoints = {doc['_id']: doc['statepoint'] for doc in project.index()}
    return pd.DataFrame(statepoints).T.join(df_index)


def df_setup(project, descriptors_filename='descriptors-h.json',
             mean=False, smiles_only=False):
    """
    """
    if not(type(project) == signac.Project):
        project = list(project)
    dfs = []
    for p in project:
        df = signac_to_df(p)
        job_ids = []
        descriptors = {}
        for i, job in enumerate(p.find_jobs()):
            df_descriptors = pd.read_json(os.path.join(job.ws, descriptors_filename), typ='series')
            if not descriptors:
                for key, val in df_descriptors.items():
                    descriptors[key] = [val]
            else:
                for key, val in df_descriptors.items():
                    descriptors[key].append(val)
            job_ids.append(job.get_id())
        for key, val in descriptors.items():
            val = [x for _, x in sorted(zip(job_ids, val))]
            df.insert(0, key, val)
        if 'terminal_group' in df:
            df = df[df['chainlength'] == 17]
        '''
        else:
            for _, row in df.iterrows():
                if(row['terminal_groups'][0] == 'amino' and
                        row['terminal_groups'][1] == 'carboxyl'):
                    print(row['hbonds'])
                    print(row['asphericity-mean'])
                    print(row['hk-kappa3-mean'])
        '''
        if smiles_only:
            keep_columns = ['COF', 'intercept',
                'terminal_group', 'terminal_groups']
        else:
            keep_columns = ['COF', 'intercept',
                'avg_molecular_weight', 'delta_molecular_weight',
                'min_molecular_weight', 'max_molecular_weight',
                'avg_dipole_moment', 'delta_dipole_moment',
                'min_dipole_moment', 'max_dipole_moment',
                'molecular_weight', 'dipole_moment',
                'terminal_group', 'terminal_groups']
        keep_columns = list(set(keep_columns) & set(df)) + \
                       df_descriptors.index.values.tolist()
        df = df[keep_columns]
        renaming = {
                'avg_molecular_weight': 'molecular-weight-mean',
                'delta_molecular_weight': 'molecular-weight-diff',
                'min_molecular_weight': 'molecular-weight-min',
                'max_molecular_weight': 'molecular-weight-max',
                'avg_dipole_moment': 'dipole-moment-mean',
                'delta_dipole_moment': 'dipole-moment-diff',
                'min_dipole_moment': 'dipole-moment-min',
                'max_dipole_moment': 'dipole-moment-max'
                }
        df.rename(index=str, columns=renaming, inplace=True)
        if 'terminal_groups' not in df:
            if not smiles_only:
                df['molecular-weight-mean'] = df['molecular_weight']
                df['molecular-weight-diff'] = 0.0
                df['molecular-weight-max'] = df['molecular_weight']
                df['molecular-weight-min'] = df['molecular_weight']
                df['dipole-moment-mean'] = df['dipole_moment']
                df['dipole-moment-diff'] = 0.0
                df['dipole-moment-max'] = df['dipole_moment']
                df['dipole-moment-min'] = df['dipole_moment']
            df['terminal_group_1'] = df['terminal_group']
            df['terminal_group_2'] = df['terminal_group']
            for descriptor in df_descriptors.index.values:
                if descriptor != 'hbonds':
                    df['{}-mean'.format(descriptor)] = df[descriptor]
                    df['{}-diff'.format(descriptor)] = 0.0
                    df['{}-max'.format(descriptor)] = df[descriptor]
                    df['{}-min'.format(descriptor)] = df[descriptor]
            if smiles_only:
                to_remove = ['terminal_group'] + df_descriptors.index.values.tolist()
            else:
                to_remove = ['dipole_moment', 'molecular_weight',
                             'terminal_group'] + df_descriptors.index.values.tolist()
            to_remove.remove('hbonds')
            df.drop(to_remove, axis=1, inplace=True)
        else:
            df['terminal_group_1'] = df['terminal_groups'].apply(lambda x: x[0])
            df['terminal_group_2'] = df['terminal_groups'].apply(lambda x: x[1])
            df.drop(['terminal_groups'], axis=1, inplace=True)
        dfs.append(df)

    df = pd.concat(dfs, sort=True)
    if mean:
        df = df.groupby(['terminal_group_1', 'terminal_group_2'],
                        as_index=False).mean()
    return df
