import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.misc import *
import re
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
import pickle
import matplotlib.patches as mpatches
from matplotlib import ticker
import matplotlib.patheffects as path_effects
from data_handling.preprocessing import DataPreprocessor
from utils.utils import error_handling_makedirs

xlabel_dict = {'train' : 'training dataset',
                'val' : 'validation dataset'}
val_test_keys = ['loss', 'pitch_precision', 'pitch_recall', 'pitch_f1',
                    'tab_precision', 'tab_recall', 'tab_f1', 'tdr', 'learning_rate']
plot_names_cat = ['Loss value', 'Pitch precision', 'Pitch recall', r'Multi-Pitch $F_1$',
                'Tablature precision', 'Tablature recall', r'Tablature $F_1$', 'TDR', 'Learning Rate']
name_mapping = {
        'CrossDataset': 'Baseline',
        'FocalLoss': 'Focal Loss',
        'DistanceWeighted': r'Distance Weighting ($w^{D}$ )',
        'SampleWeighted': r'Sample Weighting ($w^{S}$)',
        'ClassWeighted': r'Class Weighting ($w^{C}$)',
        'Oversampling': 'W/ Oversampling',
        'PreTrain1K': 'PreTrain1K',
        'PreTrain5K': 'PreTrain5K',
        'PreTrain10K': 'PreTrain10K',
        'PreTrain15K': 'PreTrain15K',
        'HodgepodgePreTrain15K': 'S\\+ PreTrain15K',
        'CrossDSWithHodgepodge15K': r'Pretrained W/ S\\+$_{15K}$',
        'CrossDSWithSynthTab1K': r'Pretrained W/ SynthTab$_{1K}$',
        'CrossDSWithSynthTab5K': r'Pretrained W/ SynthTab$_{5K}$',
        'CrossDSWithPLUS5K': r'Pretrained W/ PLUS$_{5K}$',
        'CrossDSWithSynthTab10K': r'Pretrained W/ SynthTab$_{10K}$',
        'CrossDSWithPLUS10K': r'Pretrained W/ PLUS$_{10K}$',
        'CrossDSWithSynthTab15K': r'Pretrained W/ SynthTab$_{15K}$',
        'CrossDSWithPLUS15K': r'Pretrained W/ PLUS$_{15K}$',
        'ClassWeightedWithPLUS10K': r'$w^{C}$ \& Pretrained W/ PLUS$_{10K}$',
        'ClassWeightedWithSynthTab10K': r'$w^{C}$ \& Pretrained W/ SynthTab$_{10K}$'
    }

def load_pkl_res_as_df(exp_strings):
    df = None
    for exp_str in exp_strings:
        file = open(PREV_MODELS_PATH + f'/{exp_str}/model_results.pkl', 'rb')
        results_dict = pickle.load(file)
        exp_df = pd.DataFrame(results_dict)
        exp_df['experiment'] = [exp_str] * len(exp_df) 
        df = pd.concat([df, exp_df]) if df is not None else exp_df
    df.reset_index(inplace=True)
    df['experiment'] = df['experiment'].replace(name_mapping)
    return df

def plot_anim_tablature(plot_labels, sr=22050, hop_length=512, 
                        save_fig=False, fig_name='example.mp4'):
    plot_labels= plot_labels[:,:, 1:]
    fig = plt.figure()
    fps = sr / hop_length
    im = plt.imshow(plot_labels[0], vmin=0, vmax=1, cmap='binary')
    plt.xticks(range(plot_labels.shape[-1]))
    for x in range(1, plot_labels.shape[-1]):
        plt.axvline(x - 0.5, color='gray', alpha=0.5)
    for x in [5, 7, 9, 15, 17]:
        plt.scatter(x, 2.5, color='black', alpha=0.5)
    plt.scatter([12, 12], [1.5, 3.5], color='black', alpha=0.5)
    plt.yticks(range(0,6), labels=['E', 'A', 'D', 'G', 'B', 'E'])
    annot = plt.annotate('Second 0', (0.09, 0.72), xycoords='figure fraction')
    def animate(i):
        im.set_array(plot_labels[i])
        annot.set_text(f'sec {i / fps}')
        return im, annot
    anim = FuncAnimation(fig, animate, interval=1000/fps,
                        frames=len(plot_labels),)
    if save_fig:
        error_handling_makedirs(VIS_PATH)
        anim.save(VIS_PATH + '/' + fig_name,
                  fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()

def plot_anim_pitches(spec, plot_labels, sr=22050, hop_length=512,
                      save_fig=False, fig_name='example_pitch.mp4'):
    fps = sr / hop_length
    string_pitches = torch.tensor(MIDI_NOTE_NUMBERS)
    pitches = torch.zeros(6, len(plot_labels))
    sample_nr, string, fret = torch.where(plot_labels[..., 1:] == 1)
    active_pitches = (string_pitches[string] + fret).type(dtype=torch.float32)
    pitches[string, sample_nr] = active_pitches
    ind = torch.where(pitches == 0)
    pitches[ind] = float('nan')
    pitches = (pitches - 24) * 2
    
    fig = plt.figure()
    plt.xlim([0, 200])
    plt.ylim([0, 192])
    annot = plt.annotate('Second 0', (0.09, 1), xycoords='figure fraction')
    im = plt.imshow(spec[:, :200], origin='lower',
                    vmin = float(spec.min()), vmax = float(spec.max()))
    x_vals = torch.arange(0, 200)
    sca1, = plt.plot([], [])
    sca2, = plt.plot([], [])
    sca3, = plt.plot([], [])
    sca4, = plt.plot([], [])
    sca5, = plt.plot([], [])
    sca6, = plt.plot([], [])

    def animate(i):
        im.set_array(spec[:, i:200+i])
        # Changed len(spec) to spec.shape[1] to check against total time steps N
        if i != spec.shape[1]: 
            for j, sca in enumerate([sca1, sca2, sca3, sca4, sca5, sca6,]):
                plotted_pitches = pitches[j][i:200+i]
                sca.set_data([x_vals[:len(plotted_pitches)], plotted_pitches])
                sca.set_color('red')
                sca.set_alpha(0.5)
                annot.set_text(f'sec {i / fps}')
        return im, sca1, sca2, sca3, sca4, sca5, sca6
        
    anim = FuncAnimation(fig, animate, interval= 1000/fps,
                        frames=spec.shape[1],)
    if save_fig:
        error_handling_makedirs(VIS_PATH)
        anim.save(VIS_PATH + '/' + fig_name,
                  fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()

def plot_training_results(df, save_dir):
    error_handling_makedirs(save_dir)
    # this can plot train and val. It could not plot something useful for test, which requires a bar chart.
    for i in range(len(val_test_keys)):
        cat_plot = val_test_keys[i]
        cat_plot_name = plot_names_cat[i]
        unique_loader_keys = ['val']
        if i == 0:
            unique_loader_keys = ['train', 'val']
        unique_ds_keys = df.ds_train_on.unique()
        plt.figure(figsize=(4,4))
        for idx, loader in enumerate(unique_loader_keys):
            for dataset in unique_ds_keys:
                mini_df = df[(df['loader'] == loader) * (df['ds_train_on'] == dataset)]
                subsec = df[df['loader'] != 'test']
                mini_df.reset_index(inplace=True)
                plt.subplot(len(unique_loader_keys), 1, idx + 1)
                plt.plot(mini_df[cat_plot])
                plt.grid(zorder=0, linestyle=':', axis='y')
                if i != 0:
                    plt.ylim([0., 1.05])
                else:
                    plt.ylim([0, max(subsec[cat_plot]) + 1])
                plt.title(label=f'The {cat_plot_name} per epoch for each model on the {xlabel_dict[loader]}.')
                plt.xlabel('epochs')
                plt.ylabel(cat_plot_name)
            plt.figlegend(unique_ds_keys, bbox_to_anchor=(1.35, 0.9),
                        title='Dataset on which\nmodel was trained.')
        plt.tight_layout()

        plt.savefig(save_dir + f'/{cat_plot}_val.png', bbox_inches = "tight")

def plot_test_results(df, save_dir=None, colors=None, hatches=None):
    for i in range(len(val_test_keys)):

        cat_plot = val_test_keys[i]
        cat_plot_name = plot_names_cat[i]

        if cat_plot == 'learning_rate':
            continue
        
        plt.figure(figsize=(6,12))
        mini_df = df[(df['loader']=='test')]
        pivot_df = mini_df.pivot(index='ds_train_on',
                            columns='ds_eval_on',
                            values=cat_plot)
        max_val = pivot_df.max(axis=None)
        if max_val < 1:
            pivot_df = pivot_df * 100
            max_val = 100
        ax = pivot_df.plot.bar(figsize=(12, 6), width=0.75, color=colors)

        for bar_i, container in enumerate(ax.containers):
            for bar in container:
                height = bar.get_height()
                if hatches is not None:
                    bar.set_hatch(hatches[bar_i])
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_val, f'{height:.1f}', 
                            ha='center', va='center', color='black', fontweight='bold',
                            fontsize=7)
        plt.grid(zorder=0, linestyle=':', axis='y')

        plt.ylabel(cat_plot_name)
        plt.xlabel('Models based on which dataset they were trained on.')
        plt.xticks(rotation=45)
        title_str = (f'Grouped bar chart for the {cat_plot_name} of each model and test set.\n')
        text_str = (f'Each bar chart group is for a model named after the dataset it was trained on' + 
                    'and its performance for each test set is the bar which is labeled in the legend.')
        plt.figtext(0.15, 0.9, text_str, fontsize=8)
        plt.title(title_str)
        if i != 0:
            plt.ylim([0., 100])
        plt.legend(bbox_to_anchor=(1, 1), title='Test dataset on which\nmodel was evaluated.')
        if save_dir is not None:
            error_handling_makedirs(save_dir)
            plt.savefig(save_dir + f'/{cat_plot}.png', bbox_inches = "tight")
        plt.show()

def plot_multi_exp_test_results(df, test_ds='EGSet12', save_dir=None, colors=None, hatches=None):
    for i in range(len(val_test_keys)):

        cat_plot = val_test_keys[i]
        cat_plot_name = plot_names_cat[i]

        mini_df = df[(df.ds_eval_on==test_ds) & (df.loader == 'test')]
        pivot_df = mini_df.pivot(index='ds_train_on',
                                columns='experiment',
                                    values=cat_plot)
        ax = pivot_df.plot.bar(figsize=(12, 6))

        max_val = pivot_df.max(axis=None)
        if max_val < 1:
            pivot_df = pivot_df * 100
            max_val = 100
        ax = pivot_df.plot.bar(figsize=(12, 6), width=0.75, color=colors)

        for bar_i, container in enumerate(ax.containers):
            for bar in container:
                if hatches is not None:
                    bar.set_hatch(hatches[bar_i])
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_val, f'{height:.1f}', 
                            ha='center', va='center', color='black', fontweight='bold',
                            fontsize=7)

        plt.grid(zorder=0, linestyle=':', axis='y')
        plt.ylabel(cat_plot_name)
        plt.xlabel('Models based on which dataset they were trained on.')
        plt.xticks(rotation=45)
        title_str = (f'Grouped bar chart for the {cat_plot_name} on {test_ds} of each model per experiment.\n\n')
        text_str = (f'Each bar chart group is for a dataset, from which multiple models were trained on,\n' + 
                    'each bar is a model from a specific experiment given in the legend.')
        plt.figtext(0.15, 0.9, text_str, fontsize=8)
        plt.title(title_str)
        if i != 0:
            plt.ylim([0., 100])
        plt.legend(bbox_to_anchor=(1, 1), title='Dataset on which\n model was pre-trained')
        plt.tight_layout()
        if save_dir is not None:
            error_handling_makedirs(save_dir)
            plt.savefig(save_dir + f'/{cat_plot}_test.png', bbox_inches = "tight")
        plt.show()

def plot_conf_mats(df, save_dir, test_ds='EGSet12'):
    labels = ['Total Silence', 'One Active String', 'Multiple Active Strings']
    condition = (df.ds_eval_on == test_ds) & (df.loader == 'test')
    conf_mats = df[condition][['ts', 'tsfss', 'tsfms',
                                'tssfs', 'tss', 'tssfms',
                                'tmsfs', 'tmsfss', 'tms']]
    indices = list(conf_mats.index)
    num_plots = len(conf_mats)

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        conf_mat = torch.tensor(conf_mats.iloc[i].values.astype(float)).reshape(3, 3)
        row_sums = conf_mat.sum(dim=1, keepdim=True)
        norm_conf_mat = torch.where(row_sums > 0, conf_mat / row_sums, torch.zeros_like(conf_mat))
        
        ax = axes[i]
        ax.imshow(norm_conf_mat, cmap='bone_r')
        ax.set_title(f'Trained on {df.ds_train_on[indices[i]]}')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels,
                           rotation=45)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(labels)
        if i == 0:
            ax.set_xlabel('Predicted sonority class')
            ax.set_ylabel('True sonority class')

        for r in range(3):
            for c in range(3):
                val = norm_conf_mat[r, c].item()
                # If the cell is dark (>= 0.5), use white text; otherwise, black text
                if val <= 0.05:
                    continue
                color = 'white'
                ax.text(c, r, f'{val*100:.1f}%', ha='center', va='center', color=color)
    fig.suptitle(f'Confusion matrices for each model on {test_ds}')
    plt.tight_layout()
    plt.savefig(save_dir + f'/conf_mats.png', bbox_inches = "tight")

def plot_multi_exp_conf_mats(df, save_dir, test_ds='EGSet12', experiment_order=None, dataset_order=None):
    labels = ['Total Silence', 'One Active String', 'Multiple Active Strings']

    # Enforce standard execution order if explicitly passed
    experiments = experiment_order if experiment_order is not None else df.experiment.unique().tolist()
    datasets = dataset_order if dataset_order is not None else df.ds_train_on.unique().tolist()
    
    num_horiz_plots = len(datasets)
    num_verti_plots = len(experiments)
    
    fig, axes = plt.subplots(num_verti_plots, num_horiz_plots,
                             figsize=(5 * num_horiz_plots, 4 * num_verti_plots))
    
    # Safe multi-dimensional array fallback adjustment
    if num_verti_plots == 1 and num_horiz_plots == 1:
        axes = torch.tensor([[axes]])
    elif num_verti_plots == 1:
        axes = axes[None, :]
    elif num_horiz_plots == 1:
        axes = axes[:, None]

    for i, exp_string in enumerate(experiments):
        for j, ds_string in enumerate(datasets):
            # Target exact dataset/experiment intersection to isolate cell measurements accurately
            condition = (df.ds_eval_on == test_ds) & (df.experiment == exp_string) & \
                        (df.ds_train_on == ds_string) & (df.loader == 'test')
            
            sub_df = df[condition]
            ax = axes[i, j]
            
            if sub_df.empty:
                # Safe visual padding empty handler if an experiment variant is missing
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                continue
                
            conf_mats = sub_df[['ts', 'tsfss', 'tsfms',
                                'tssfs', 'tss', 'tssfms',
                                'tmsfs', 'tmsfss', 'tms']]
            
            # Extract configuration matrix 
            conf_mat = torch.tensor(conf_mats.iloc[0].values.astype(float)).reshape(3, 3)
            row_sums = conf_mat.sum(dim=1, keepdim=True)
            
            # Normalize and scale to percentage scale immediately
            norm_conf_mat = torch.where(row_sums > 0, (conf_mat / row_sums) * 100, torch.zeros_like(conf_mat))

            # FIXED: Vmin/Vmax now lines up perfectly with normalized percentage indices
            im = ax.imshow(norm_conf_mat, cmap='bone_r', vmin=20, vmax=100)
            
            if i == 0:
                ax.set_title(f'Trained on {ds_string}', fontsize=12, fontweight='bold', pad=10)
                
            if j == 0:
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_ylabel(exp_string, fontsize=12, fontweight='bold', 
                              rotation=0, ha='right', va='center', labelpad=15)
            else:
                ax.get_yaxis().set_visible(False)
                
            if i == num_verti_plots - 1:
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            else:
                ax.get_xaxis().set_visible(False)
                
            # Text cell rendering
            for r in range(3):
                for c in range(3):
                    val = norm_conf_mat[r, c].item()
                    if val <= 5.0:  # Skip drawing text for low values
                        continue
                    # Color inversion check depending on heat map depth
                    color = 'white' if val >= 65.0 else 'black'
                    ax.text(c, r, f'{val:.1f}%', ha='center', va='center', color=color, fontweight='bold')
                    
    fig.suptitle(f'Confusion matrices for each model when predicting on {test_ds}', 
                 fontsize=14, fontweight='bold', y=0.98)
    fig.supxlabel('Predicted sonority class', fontsize=12, fontweight='bold')
    fig.supylabel('True sonority class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'conf_mats_multi_exp.png'), bbox_inches="tight", dpi=300)
        
    plt.show()

def plot_multi_exp_classification_metrics(df, test_ds='EGSet12', save_dir=None, colors=None, hatches=None, 
                                     dataset_order=None, experiment_order=None):
    """
    Plots a wide, 2-panel side-by-side stacked bar chart comparing Precision and Recall 
    in a single row to maximize page real estate for academic papers.
    Allows passing explicit lists for dataset_order (X-axis) and experiment_order (bars/legend ordering).
    Legends are placed horizontally at the top to optimize text-width usage.
    """
    # Removed F1 Score completely to focus narrative on Precision vs Recall
    metric_pairs = [
        ('pitch_precision', 'tab_precision', 'Precision'),
        ('pitch_recall', 'tab_recall', 'Recall')
    ]
    
    mini_df = df[(df.ds_eval_on == test_ds) & (df.loader == 'test')]
    
    # ADJUSTED ASPECT RATIO: 11x6.2 provides a perfect balanced window for a 2-panel array
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    for i, (pitch_key, tab_key, metric_label) in enumerate(metric_pairs):
        ax = axes[i]
        
        # Pivot Pitch and Tablature data
        pivot_pitch = mini_df.pivot(index='ds_train_on', columns='experiment', values=pitch_key)
        pivot_tab = mini_df.pivot(index='ds_train_on', columns='experiment', values=tab_key)
        
        # Enforce user-defined row (X-axis) and column (Experiment bars) ordering cleanly
        if dataset_order is not None:
            pivot_pitch = pivot_pitch.reindex(index=dataset_order)
            pivot_tab = pivot_tab.reindex(index=dataset_order)
        if experiment_order is not None:
            pivot_pitch = pivot_pitch.reindex(columns=experiment_order)
            pivot_tab = pivot_tab.reindex(columns=experiment_order)
        
        # Convert to native PyTorch tensors immediately to handle safe filtering
        pitch_tensor = torch.tensor(pivot_pitch.values, dtype=torch.float32)
        tab_tensor = torch.tensor(pivot_tab.values, dtype=torch.float32)
        
        # Robust scalar check to guarantee 0-100 scaling
        valid_pitch = pitch_tensor[~torch.isnan(pitch_tensor)]
        if valid_pitch.numel() > 0 and valid_pitch.max().item() <= 1.0:
            pitch_tensor = pitch_tensor * 100
            tab_tensor = tab_tensor * 100
            pivot_pitch = pivot_pitch * 100  
            
        # Generate the background container plot on the specific ax subplot
        pivot_pitch.plot.bar(ax=ax, width=0.7, color=colors, alpha=0.4, legend=False) 
        
        # Loop through containers to construct custom over-stacked layers
        for idx, (container_pitch, container_tab) in enumerate(zip(ax.containers, ax.containers)):
            current_color = container_pitch[0].get_facecolor()
            
            for bar_p, bar_t, val_p, val_t in zip(container_pitch, container_tab, pitch_tensor[:, idx], tab_tensor[:, idx]):
                if torch.isnan(val_p) or torch.isnan(val_t):
                    continue
                
                # Draw the inner/lower Tablature bar
                rect_tab = plt.Rectangle(
                    (bar_p.get_x(), 0), bar_p.get_width(), val_t.item(),
                    facecolor=current_color, edgecolor='black', alpha=1.0, linewidth=0.1
                )
                if hatches is not None:
                    rect_tab.set_hatch(hatches[idx])
                ax.add_patch(rect_tab)
                
                # HALO EFFECT SETUP: Clean stroke paths that follow character shapes directly
                pitch_stroke = [path_effects.withStroke(linewidth=2.0, foreground='white')]
                tab_stroke = [path_effects.withStroke(linewidth=1.5, foreground=current_color)]
                
                # Add Text Label for outer bounds (Pitch) - Black text with a clean white stroke glow
                ax.text(bar_p.get_x() + bar_p.get_width()/2., val_p.item() + 1.5, f'{val_p.item():.1f}', 
                        ha='center', va='bottom', color='black', fontweight='bold', fontsize=8.0, rotation=90,
                        path_effects=pitch_stroke)
                
                # Add Text Label inside inner bounds (Tablature) - White text with a complementary colored glow
                if val_t.item() > 15: 
                    x_pos = bar_p.get_x() + bar_p.get_width()/2.
                    y_pos = val_t.item() - 2.5
                    ax.text(x_pos, y_pos, f'{val_t.item():.1f}', 
                            ha='center', va='top', color='white', fontweight='bold', fontsize=8.0, rotation=90,
                            path_effects=tab_stroke)

        # Apply specific styling constraints per subplot
        ax.grid(zorder=0, linestyle=':', axis='y')
        ax.set_xticklabels(pivot_pitch.index.tolist(), rotation=45, ha="right", fontsize=10)
        ax.set_xlabel('') 
        ax.set_title(metric_label, fontweight='bold', fontsize=13, pad=10)
        ax.set_ylim([0, 100])  

    # Forced display on left plot using explicit ticks
    y_ticks = [0, 20, 40, 60, 80, 100]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels([f"{y}%" for y in y_ticks], fontsize=10)
    axes[0].yaxis.set_tick_params(labelleft=True)  
    axes[0].set_ylabel("Performance Score", fontweight='bold', fontsize=12)
            
    # Set a single, unified X-axis label across the entire figure
    fig.supxlabel('Training-sets used', y=0, fontweight='bold', fontsize=12)
    
    # --- HORIZONTAL LEGENDS CONFIGURATION AT THE TOP ---
    handles, labels = axes[-1].get_legend_handles_labels()
    
    # 1. Main Legend: Experiment Groups (Set horizontally using ncol)
    main_legend = fig.legend(handles, labels, bbox_to_anchor=(0.4, 1.05), loc='upper center', 
                             ncol=len(labels), title='Experiment Group', title_fontsize=11, fontsize=10)
    
    # Restore true color opacity to main legend icons
    if main_legend is not None:
        for lh in main_legend.legend_handles:
            lh.set_alpha(1.0)
            
    # 2. Text-Explicit Metric Layer Legend (Set horizontally using ncol=2)
    pitch_patch = mpatches.Patch(fill=False, edgecolor='none', label=r'Translucent $\rightarrow$ Multi-Pitch')
    tab_patch = mpatches.Patch(fill=False, edgecolor='none', label=r'Opaque $\rightarrow$ Tablature')
    
    fig.legend(handles=[pitch_patch, tab_patch], bbox_to_anchor=(0.85, 1.05), loc='upper center', 
               ncol=2, title='Layer Definitions', title_fontsize=11, fontsize=10, handlelength=0) 
            
    # Adjust subplots layout bounding area to clear space for the upper legends row
    plt.tight_layout()
    fig.subplots_adjust(top=0.81, bottom=0.16)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'multi_exp_scores.pdf')
        plt.savefig(save_path, bbox_inches="tight")
        
    plt.show()

def plot_multi_exp_boxplots(df, save_dir=None, experiment_order=None):
    """
    Plots a 2x1 column of vertical paired metrics.
    Top Panel: Recall vs Experiment Type
    Bottom Panel: Precision vs Experiment Type
    Uses deterministic, fixed offsets per dataset to handle overlaps elegantly.
    """
    mini_df = df[df['loader'] == 'test'].copy()
    
    metrics_config = [
        ('tab_recall', 'Recall'),
        ('tab_precision', 'Precision')
    ]
    
    for m_key, _ in metrics_config:
        if mini_df[m_key].max() <= 1.0:
            mini_df[m_key] = mini_df[m_key] * 100
            
    mini_df['is_cross_ds'] = mini_df['ds_train_on'] != mini_df['ds_eval_on']
    
    if experiment_order is None:
        experiment_order = mini_df['experiment'].unique().tolist()
    
    # --- CONFIGURATION: Mapping Colors, Shapes, and Deterministic Offsets ---
    # GOAT and IDMT stay perfectly centered (offset = 0.0)
    # The others are cleanly spread out sequentially between -0.05 and 0.05
    dataset_configs = {
        'IDMT':        {'color': '#e67e22', 'marker': 'h', 'offset':  0.0},   
        'GuitarSet':   {'color': '#9b59b6', 'marker': 'h', 'offset': 0.0},   
        'EGDB':        {'color': '#2ecc71', 'marker': 'h', 'offset':  0.0},   
        'GuitarTECHS': {'color': '#f1c40f', 'marker': 'h', 'offset': 0.0},  
        'GOAT':        {'color': '#e74c3c', 'marker': 'h', 'offset':  0.0}    
    }
    color_cross = '#2c3e50'       
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    
    v_pad = 1.25  
    
    for i, (m_key, m_title) in enumerate(metrics_config):
        ax = axes[i]
        
        current_pos = 1.0
        ticks_positions = []
        labels_list = []
        
        for exp in experiment_order:
            exp_df = mini_df[mini_df['experiment'] == exp]
            if exp_df.empty:
                continue
                
            within_df = exp_df[~exp_df['is_cross_ds']].dropna(subset=[m_key])
            cross_scores = exp_df[exp_df['is_cross_ds']][m_key].dropna().values
            
            pos_w = current_pos - 0.175
            pos_c = current_pos + 0.175
            
            # --- WITHIN DATASET PLOT (FIXED OFFSET DISTINCTION) ---
            if not within_df.empty:
                for ds_name, row in within_df.iterrows():
                    ds_label = row['ds_eval_on']
                    
                    # Pull configurations
                    cfg = dataset_configs.get(ds_label, {'color': '#7f8c8d', 'marker': 'o', 'offset': 0.0})
                    dot_color = cfg['color']
                    dot_marker = cfg['marker']
                    fixed_offset = cfg['offset']
                    
                    score_val = row[m_key]

                    # Instead of random.uniform, apply the clean fixed offset directly
                    ax.scatter(pos_w + fixed_offset, score_val, color=dot_color, marker=dot_marker, 
                               edgecolor='black', linewidth=0.4, alpha=0.9, s=28, zorder=3)
                           
            # --- CROSS DATASET PLOT & LABELS ---
            if len(cross_scores) > 0:
                ax.boxplot(cross_scores, positions=[pos_c], widths=0.28, patch_artist=True,
                           vert=True, showfliers=False, whis=(0, 100),
                           boxprops=dict(facecolor=color_cross, color='black', linewidth=0.8),
                           medianprops=dict(color='white', linewidth=1.2),
                           whiskerprops=dict(color='black', linewidth=0.8),
                           capprops=dict(color='black', linewidth=0.8))
                
                c_min, c_max = cross_scores.min(), cross_scores.max()
                
                ax.text(pos_c, c_min - v_pad, f'{c_min:.1f}%', ha='center', va='top', 
                        fontsize=6.5, color='black', fontweight='semibold', rotation=0)
                ax.text(pos_c, c_max + v_pad, f'{c_max:.1f}%', ha='center', va='bottom', 
                        fontsize=6.5, color='black', fontweight='semibold', rotation=0)
                
            ticks_positions.append(current_pos)
            labels_list.append(str(exp))
            current_pos += 1.0
            
        ax.set_ylabel(m_title, fontsize=10, fontweight='bold')
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=8.5)
        
        if ticks_positions:
            ax.set_xlim([0.5, ticks_positions[-1] + 0.5])
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1.0, offset=0.5))
        
        ax.grid(axis='y', which='major', linestyle=':', alpha=0.5, zorder=0)
        ax.grid(axis='x', which='minor', linestyle=':', color='gray', alpha=0.4, zorder=0)
        ax.set_ylim([-3.25, 103.25])  
        
    axes[1].set_xticks(ticks_positions)
    axes[1].set_xticklabels(labels_list, fontsize=9, fontweight='bold', rotation=15, ha='right')
    axes[1].set_xlabel(r'Within-\ \& Cross-Dataset Results per Experiment Type', fontsize=10, fontweight='bold', labelpad=6)
    
    # --- LEGEND ARCHITECTURE ---
    legend_handles = []
    legend_labels = []
    
    # 1. Header for Within Dataset group
    legend_handles.append(mpatches.Patch(fill=False, edgecolor='none'))
    legend_labels.append('Within Dataset:')
    
    # 2. Unique within-dataset markers
    for ds_name, cfg in dataset_configs.items():
        scatter_handle = plt.Line2D([0], [0], marker=cfg['marker'], color='w', markerfacecolor=cfg['color'], 
                                    markeredgecolor='black', markeredgewidth=0.4, markersize=6.0)
        legend_handles.append(scatter_handle)
        legend_labels.append(ds_name)
        
    # 3. Spacer element
    legend_handles.append(mpatches.Patch(fill=False, edgecolor='none'))
    legend_labels.append('   ')
    
    # 4. Cross-Dataset start text block
    cross_start = mpatches.Patch(facecolor=color_cross, edgecolor='black', linewidth=0.8)
    legend_handles.append(cross_start)
    legend_labels.append('Cross-Dataset (Agg. over ')
    
    # 5. Pack the shapes inside the tuple tightly
    sub_dots_list = []
    for ds_name, cfg in dataset_configs.items():
        sub_dot = plt.Line2D([0], [0], marker=cfg['marker'], color='w', markerfacecolor=cfg['color'], 
                             markeredgecolor='black', markeredgewidth=0.4, markersize=6.0)
        sub_dots_list.append(sub_dot)
    
    legend_handles.append(tuple(sub_dots_list))
    legend_labels.append(')')  
    
    fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.95), ncol=9, fontsize=8.5, frameon=True,
               columnspacing=0.5, handletextpad=0.3,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, hspace=0.05)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'multi_exp_boxplots.pdf'), bbox_inches='tight')
        
    plt.show()

def plot_varying_ds_sizes(df, eval_ds_list=['EGDB', 'EGSet12'], save_dir=None):
    """
    Plots a multi-row grid of line charts tracking metric progression.
    Merges X-ticks column-wise, merges Y-ticks row-wise using a dynamic grid 
    divisible by 5, alternates dot text by checking relative line heights,
    and tightens horizontal layout space.
    """
    mini_df = df[df.ds_eval_on.isin(eval_ds_list) & (df.loader == 'test')].copy()
    
    def extract_size(exp_name):
        match = re.search(r'PreTrain(\d+)K', str(exp_name), re.IGNORECASE)
        return int(match.group(1)) if match else None

    mini_df['size_int'] = mini_df['experiment'].apply(extract_size)
    mini_df = mini_df.dropna(subset=['size_int'])
    
    detected_sizes = sorted(mini_df['size_int'].unique().astype(int).tolist())
    if not detected_sizes:
        print("Warning: No valid PreTrain{n}K experiment sizes detected.")
        return

    size_to_ordinal = {size: idx for idx, size in enumerate(detected_sizes)}
    mini_df['x_ordinal'] = mini_df['size_int'].map(size_to_ordinal)
    
    metrics_config = [
        ('tab_precision', 'Tablature Precision'),
        ('tab_recall', 'Tablature Recall')
    ]
    
    for m_key, _ in metrics_config:
        if mini_df[m_key].max() <= 1.0:
            mini_df[m_key] = mini_df[m_key] * 100

    targets = ['SynthTab', 'PLUS']
    colors = {
        'SynthTab': '#e67e22',
        'PLUS': '#2c3e50'
    }
    
    num_rows = len(eval_ds_list)
    num_cols = len(metrics_config)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7, 2 * num_rows), 
                             sharex='col', sharey='row')
    
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    for r_idx, eval_ds in enumerate(eval_ds_list):
        ds_filtered_df = mini_df[mini_df.ds_eval_on == eval_ds]
        
        # --- CALCULATE ROW-WISE Y-LIMITS DIVISIBLE BY 5 ---
        row_data = ds_filtered_df[ds_filtered_df['ds_train_on'].isin(targets)][[m[0] for m in metrics_config]].dropna()
        if not row_data.empty:
            global_min = row_data.values.min()
            global_max = row_data.values.max()
            
            # Floor down and ceil up to the nearest multiple of 5 (with a tighter 1.5% margin)
            y_min = int(np.floor((global_min - 1.5) / 5.0) * 5)
            y_min = max(0, y_min)
            
            y_max = int(np.ceil((global_max + 1.5) / 5.0) * 5)
            y_max = min(100, y_max)
            
            y_ticks = list(range(y_min, y_max + 1, 5))
        else:
            y_min, y_max, y_ticks = 0, 100, [0, 20, 40, 60, 80, 100]

        for c_idx, (m_key, m_title) in enumerate(metrics_config):
            ax = axes[r_idx, c_idx]
            
            # Extract line vectors to compute dynamically who is on top per tick mark
            lines_data = {}
            for t_ds in targets:
                line_df = ds_filtered_df[ds_filtered_df['ds_train_on'] == t_ds].sort_values('size_int')
                if not line_df.empty:
                    lines_data[t_ds] = line_df
            
            # Reconstruct and plot
            for t_ds in targets:
                if t_ds not in lines_data:
                    continue
                ldf = lines_data[t_ds]
                x_vals = ldf['x_ordinal'].values
                y_vals = ldf[m_key].values
                
                ax.plot(x_vals, y_vals, color=colors[t_ds], 
                        linewidth=1.5, marker='o', markersize=4, label=t_ds)
                
                # --- DYNAMIC PLACEMENT BY REAL-TIME HEIGHT ---
                for idx, (x, y) in enumerate(zip(x_vals, y_vals)):
                    # Check if another line exists at this specific coordinate index to compare heights
                    other_targets = [t for t in targets if t != t_ds and t in lines_data]
                    
                    is_top = True
                    if other_targets:
                        other_df = lines_data[other_targets[0]]
                        if idx < len(other_df):
                            other_y = other_df[m_key].values[idx]
                            # If our value is smaller, or if equal we default SynthTab to top to break ties cleanly
                            if y < other_y or (y == other_y and t_ds == 'PLUS'):
                                is_top = False
                    
                    # Apply offsets dynamically depending on actual height relationships
                    offset = 0.8 if is_top else -0.8
                    va = 'bottom' if is_top else 'top'
                    
                    ax.text(x, y + offset, f'{y:.1f}%', ha='center', va=va, 
                            fontsize=7, color='black', fontweight='semibold')
            
            if r_idx == 0:
                ax.set_title(m_title, fontsize=9, fontweight='bold', pad=10)
                
            if c_idx == 0:
                ax.set_ylabel(f'Score on {eval_ds}', fontsize=8, fontweight='bold')
            
            ax.set_ylim(y_min - 1, y_max + 1)
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
            ax.tick_params(axis='y', labelsize=8)
            
            ax.set_xticks(list(size_to_ordinal.values()))
            ax.set_xticklabels([f'{s}K' for s in detected_sizes], fontsize=8)
            
            ax.grid(axis='both', linestyle=':', alpha=0.5)
            ax.set_xlim(-0.3, len(detected_sizes) - 0.7)
        
    fig.text(0.5, 0.04, 'Number of Audio Files Used in Training (in Thousands)', 
             ha='center', fontsize=8.5, fontweight='bold')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset used in pretraining:", title_fontsize=8.5,
               loc="upper center", bbox_to_anchor=(0.55, 0.87), ncol=2, fontsize=8,)
    
    plt.tight_layout()
    # CHANGED: wspace dropped down significantly to bring columns together seamlessly
    fig.subplots_adjust(top=0.84 - (0.04 * num_rows), bottom=0.14, wspace=0.06, left=0.12, right=0.96)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'synthetic_ds_scaling.pdf'), bbox_inches='tight')
    plt.show()

def plot_multi_exp_f1(df, test_ds='EGSet12', save_dir=None, colors=None, hatches=None, 
                      dataset_order=None, experiment_order=None):
    """
    Plots a single-panel stacked bar chart tracking Pitch and Tablature F1 score.
    Uses absolute figure legend positioning to guarantee elements never cut off.
    """
    pitch_key = 'pitch_f1'
    tab_key = 'tab_f1'
    
    mini_df = df[(df.ds_eval_on == test_ds) & (df.loader == 'test')]
    
    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    
    pivot_pitch = mini_df.pivot(index='ds_train_on', columns='experiment', values=pitch_key)
    pivot_tab = mini_df.pivot(index='ds_train_on', columns='experiment', values=tab_key)
    
    if dataset_order is not None:
        pivot_pitch = pivot_pitch.reindex(index=dataset_order)
        pivot_tab = pivot_tab.reindex(index=dataset_order)
    if experiment_order is not None:
        pivot_pitch = pivot_pitch.reindex(columns=experiment_order)
        pivot_tab = pivot_tab.reindex(columns=experiment_order)
    
    pitch_tensor = torch.tensor(pivot_pitch.values, dtype=torch.float32)
    tab_tensor = torch.tensor(pivot_tab.values, dtype=torch.float32)
    
    valid_pitch = pitch_tensor[~torch.isnan(pitch_tensor)]
    if valid_pitch.numel() > 0 and valid_pitch.max().item() <= 1.0:
        pitch_tensor = pitch_tensor * 100
        tab_tensor = tab_tensor * 100
        pivot_pitch = pivot_pitch * 100  
        
    pivot_pitch.plot.bar(ax=ax, width=0.7, color=colors, alpha=0.4, legend=False) 
    
    for idx, (container_pitch, container_tab) in enumerate(zip(ax.containers, ax.containers)):
        current_color = container_pitch[0].get_facecolor()
        
        for bar_p, bar_t, val_p, val_t in zip(container_pitch, container_tab, pitch_tensor[:, idx], tab_tensor[:, idx]):
            if torch.isnan(val_p) or torch.isnan(val_t):
                continue
            
            rect_tab = plt.Rectangle(
                (bar_p.get_x(), 0), bar_p.get_width(), val_t.item(),
                facecolor=current_color, edgecolor='black', alpha=1.0, linewidth=0.1
            )
            if hatches is not None:
                rect_tab.set_hatch(hatches[idx])
            ax.add_patch(rect_tab)
            
            pitch_stroke = [path_effects.withStroke(linewidth=2.0, foreground='white')]
            tab_stroke = [path_effects.withStroke(linewidth=1.5, foreground=current_color)]
            
            ax.text(bar_p.get_x() + bar_p.get_width()/2., val_p.item() + 1.5, f'{val_p.item():.1f}', 
                    ha='center', va='bottom', color='black', fontweight='bold', fontsize=8.0, rotation=0,
                    path_effects=pitch_stroke)
            
            if val_t.item() > 15: 
                x_pos = bar_p.get_x() + bar_p.get_width()/2
                y_pos = val_t.item() - 2.5
                ax.text(x_pos, y_pos, f'{val_t.item():.1f}', 
                        ha='center', va='top', color='white', fontweight='bold', fontsize=8.0, rotation=0,
                        path_effects=tab_stroke)

    # Grid and basic layout
    ax.grid(zorder=0, linestyle=':', axis='y')
    ax.set_xticklabels(pivot_pitch.index.tolist(), rotation=20, ha="right", fontsize=10)
    ax.set_xlabel('training-set used to train the model', fontweight='bold', fontsize=11, labelpad=4) 
    ax.set_ylim([0, 100])  

    y_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}%" for y in y_ticks], fontsize=10)
    ax.set_ylabel("Multi-Pitch \\& Tab. $F_1$ Score", fontweight='bold', fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    
    fig.subplots_adjust(top=0.78, bottom=0.28, left=0.11, right=0.96)
    
    # 3. Experiment Group Legend (Top Left)
    legend_left = fig.legend(
        handles, labels, 
        bbox_to_anchor=(0.175, 0.78), loc='lower left', 
        ncol=3, title='Experiment Type', title_fontsize=9.5, fontsize=8.5,
        frameon=True
    )
    if legend_left is not None:
        for lh in legend_left.legend_handles:
            lh.set_alpha(1.0)
            
    # 4. Layer Definitions Legend (Bottom Left Corner)
    pitch_patch = mpatches.Patch(fill=False, edgecolor='none', label=r'Translucent: Multi-Pitch')
    tab_patch = mpatches.Patch(fill=False, edgecolor='none', label=r'Opaque: Tablature')
    
    fig.legend(
        handles=[pitch_patch, tab_patch], 
        bbox_to_anchor=(0.05, 0.07), loc='lower left', 
        ncol=2, title='Layer Definitions', title_fontsize=9.5, fontsize=8.5,
        columnspacing=1.0, handlelength=0, frameon=True
    )
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'multi_exp_f1_scores.pdf')
        # bbox_inches="tight" now preserves exact constraints cleanly
        plt.savefig(save_path, bbox_inches="tight")
        
    plt.show()

def plot_test_matrices(df, save_dir=None):
    for i in range(len(val_test_keys)):
        cat_plot = val_test_keys[i]
        cat_plot_name = plot_names_cat[i]

        if cat_plot in ['learning_rate', 'loss']:
            continue
        
        mini_df = df[df['loader'] == 'test']
        mini_df = mini_df[['ds_eval_on', 'ds_train_on', cat_plot]]
        
        matrix_df = mini_df.pivot(index='ds_train_on', columns='ds_eval_on', values=cat_plot)
        
        max_val = matrix_df.max(axis=None)
        if max_val <= 1.0:
            matrix_df = matrix_df * 100
            
        matrix_df = matrix_df.reindex(
            index=['IDMT', 'GuitarSet', 'EGDB', 'GuitarTECHS', 'GOAT'],
            columns=['IDMT', 'GuitarSet', 'EGDB', 'GuitarTECHS', 'GOAT', 'EGSet12']
        )
        
        data_matrix = torch.tensor(matrix_df.values, dtype=torch.float32)
        row_categories = matrix_df.index.tolist()  
        col_categories = matrix_df.columns.tolist()  
        
        fig, ax = plt.subplots(figsize=(7.5, 5.5))  # Adjusted slightly for right-side colorbar padding
        
        im = ax.imshow(data_matrix, cmap='bone_r', vmin=20, vmax=100)
        ax.set_aspect('equal')
        
        # --- VERTICAL COLORBAR ON THE RIGHT ---
        # The title text is now elegantly integrated as the colorbar label
        cbar = ax.figure.colorbar(
            im, ax=ax, orientation='vertical', pad=0.04, shrink=0.8
        )
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(
            f"{cat_plot_name}", 
            fontsize=9, fontweight='bold', labelpad=12
        )
        
        ax.set_xticks(torch.arange(len(col_categories)))
        ax.set_yticks(torch.arange(len(row_categories)))
        ax.set_xticklabels(col_categories, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(row_categories, fontsize=9)
        
        nan_mask = torch.isnan(data_matrix)
        if not nan_mask.all():
            global_max = data_matrix[~nan_mask].max()
        else:
            global_max = torch.tensor(100.0)
        
        for r in range(len(row_categories)):
            for c in range(len(col_categories)):
                if not torch.isnan(data_matrix[r, c]):
                    text_color = "w" if data_matrix[r, c] > (global_max / 2) else "k"
                    
                    ax.text(c - 0.325, r - 0.325, f"{data_matrix[r, c].item():.1f}", 
                            ha="center", va="center", 
                            color=text_color, fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Training-set used', fontsize=9, fontweight='bold')
        ax.set_xlabel('Test-set used', fontsize=9, fontweight='bold')
        
        fig.tight_layout()
        
        if save_dir is not None:
            error_handling_makedirs(save_dir)
            save_path = os.path.join(save_dir, f'matrix_{cat_plot}.pdf')
            plt.savefig(save_path, bbox_inches="tight")
            
        plt.show()
 