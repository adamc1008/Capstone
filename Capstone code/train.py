import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, deque
import json
from datetime import datetime
import time

import gym
import gym_malware
from gym_malware.envs.utils import interface, pefeatures
from gym_malware.envs.controls import manipulate2 as manipulate
from gym_malware import sha256_holdout, MAXTURNS

from train_agent_chainer import create_acer_agent
import chainerrl

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_FAMILY_MAP = None

def load_family_map(csv_path="gym_malware/gym_malware/envs/utils/sample_hashes.csv"):
    """Load family mapping from CSV file."""
    global _FAMILY_MAP
    if '_FAMILY_MAP' not in globals() or _FAMILY_MAP is None:
        try:
            df = pd.read_csv(csv_path)
            df["Filename"] = df["Filename"].astype(str).str.lower()
            df["Subfolder"] = df["Subfolder"].astype(str)
            _FAMILY_MAP = dict(zip(df["Filename"], df["Subfolder"]))
        except Exception as e:
            _FAMILY_MAP = {}
    return _FAMILY_MAP

FULL_ACTION_TABLE = {
    'overlay_append': 'overlay_append',
    'imports_append': 'imports_append', 
    'section_rename': 'section_rename',
    'section_add': 'section_add',
    'section_append': 'section_append',
    'create_new_entry': 'create_new_entry',
    'remove_signature': 'remove_signature',
    'remove_debug': 'remove_debug',
    'upx_pack': 'upx_pack',
    'upx_unpack': 'upx_unpack',
    'break_optional_header_checksum': 'break_optional_header_checksum',
}

class OptimizedScoreBasedMalwareEnv(gym.Env):
    def __init__(self, action_list, success_threshold=0.998, 
                 reward_scale=1000, reward_clip=100, max_turns=20):
        super().__init__()
        
        self.action_list = action_list
        self.action_space = gym.spaces.Discrete(len(action_list))
        self.success_threshold = success_threshold
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.max_turns = max_turns

        self.pe_extractor = pefeatures.PEFeatureExtractor()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.pe_extractor.dim,)
        )

        self.score_baseline = 0.5
        self.baseline_decay = 0.99
        self.episode_scores = deque(maxlen=100)

        self.episode_count = 0
        self.action_stats = defaultdict(lambda: {'used': 0, 'success': 0, 'avg_improvement': 0.0})
        self.family_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0, 'avg_score': 0.0})
        
        print(f"Created environment with {len(action_list)} actions: {action_list}")
        
    def reset(self):
        sha256 = np.random.choice(sha256_holdout)
        self.bytez = interface.fetch_file(sha256)
        self.sha256 = sha256

        self.original_score = interface.get_score_local(self.bytez)
        self.current_score = self.original_score
        self.turn = 0
        self.episode_count += 1

        self.malware_family = self._guess_malware_family(sha256)
        
        return self._get_observation()
    
    def _guess_malware_family(self, sha256):
        family_map = load_family_map()

        if sha256.lower() in family_map:
            return family_map[sha256.lower()]

        for prefix_len in [8, 16, 32]:
            prefix = sha256.lower()[:prefix_len]
            for filename, family in family_map.items():
                if filename.startswith(prefix):
                    return family

        try:
            features = self.pe_extractor.extract(self.bytez)
            
            if features[0] > 0.8:
                return "packed_unknown"
            elif features[10] > 0.7:  
                return "trojan_unknown"
            elif features[20] < 0.3:  
                return "virus_unknown" 
            elif self.original_score > 0.999:
                return "advanced_unknown"
            else:
                return "generic_unknown"
                
        except:
            return "unknown"
    
    def step(self, action):
        action_name = self.action_list[action]
        
        try:
            self.bytez = manipulate.modify_without_breaking(self.bytez, [action_name])
            self.current_score = interface.get_score_local(self.bytez)
            
            self.action_stats[action_name]['used'] += 1
            
        except Exception as e:
            reward = -0.1
            info = {'error': str(e), 'action': action_name, 'final_score': self.current_score}
            return self._get_observation(), reward, False, info
        
        reward = self._calculate_optimized_reward(action_name)

        success = self.current_score < self.success_threshold
        self.turn += 1
        done = success or (self.turn >= self.max_turns)

        improvement = self.original_score - self.current_score
        self.action_stats[action_name]['avg_improvement'] = (
            (self.action_stats[action_name]['avg_improvement'] * 
             (self.action_stats[action_name]['used'] - 1) + improvement) /
            self.action_stats[action_name]['used']
        )
        
        if success:
            self.action_stats[action_name]['success'] += 1
            self.family_stats[self.malware_family]['successes'] += 1

        if done:
            self.episode_scores.append(self.current_score)
            self.family_stats[self.malware_family]['attempts'] += 1
            self.family_stats[self.malware_family]['avg_score'] = (
                (self.family_stats[self.malware_family]['avg_score'] * 
                 (self.family_stats[self.malware_family]['attempts'] - 1) + self.current_score) /
                self.family_stats[self.malware_family]['attempts']
            )
            
            self._update_baseline()
        
        info = {
            'original_score': self.original_score,
            'current_score': self.current_score,
            'improvement': improvement,
            'success': success,
            'action': action_name,
            'turn': self.turn,
            'reward': reward,
            'family': self.malware_family,
            'baseline': self.score_baseline,
            'final_score': self.current_score
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_optimized_reward(self, action_name):
        improvement = self.original_score - self.current_score
        
        if self.current_score < self.success_threshold:
            return self.reward_clip  

        baseline_improvement = improvement - (self.original_score - self.score_baseline)
        score_reward = baseline_improvement * self.reward_scale

        efficiency_bonus = max(0, (self.max_turns - self.turn) / self.max_turns) * 10

        if improvement > 0.001:  
            progress_bonus = 20
        elif improvement > 0.0001: 
            progress_bonus = 5
        else:
            progress_bonus = 0
        
        total_reward = score_reward + efficiency_bonus + progress_bonus
        
        total_reward = np.clip(total_reward, -self.reward_clip, self.reward_clip)
        
        return total_reward
    
    def _update_baseline(self):
        if len(self.episode_scores) > 10:
            recent_mean = np.mean(list(self.episode_scores)[-10:])
            self.score_baseline = (self.baseline_decay * self.score_baseline + 
                                 (1 - self.baseline_decay) * recent_mean)
    
    def _get_observation(self):
        features = self.pe_extractor.extract(self.bytez)
        return features.astype(np.float32)
    
    def get_action_stats(self):
        return dict(self.action_stats)
    
    def get_family_stats(self):
        return dict(self.family_stats)


def create_action_spaces():
    action_spaces = {
        "full_space": {
            "name": "Full Action Space",
            "actions": list(FULL_ACTION_TABLE.keys()),
            "description": "All 11 available actions"
        },
        
        "multi_step": {
            "name": "Multi-Step Optimized", 
            "actions": [
                'remove_signature',     
                'remove_debug',         
                'section_rename',        
                'overlay_append',      
                'upx_pack',            
                'break_optional_header_checksum'  
            ],
            "description": "Optimized sequence for multi-step evasion"
        },
        
        "strategic_best": {
            "name": "Strategic Best",
            "actions": [
                'remove_signature',     
                'upx_pack',            
                'section_rename',       
                'overlay_append',       
                'remove_debug',        
                'imports_append',      
                'break_optional_header_checksum'  
            ],
            "description": "Claude's strategic recommendation based on ML evasion research"
        },
        
        "half_space": {
            "name": "Half Action Space",
            "actions": [
                'remove_signature',     
                'upx_pack',            
                'section_rename',      
                'overlay_append',      
                'remove_debug',         
                'imports_append'       
            ],
            "description": "Top 6 most effective actions"
        }
    }
    
    return action_spaces


def validate_action_space(space_name, space_config, num_runs=10): 
    try:
        env = OptimizedScoreBasedMalwareEnv(
            action_list=space_config['actions'],
            success_threshold=0.998,
            reward_scale=1000,
            reward_clip=100,
            max_turns=25
        )
        
        rewards = []
        scores = []
        action_counts = defaultdict(int)
        
        for episode in range(num_runs):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = np.random.choice(len(space_config['actions']))
                state, reward, done, info = env.step(action)
                episode_reward += reward
                action_counts[space_config['actions'][action]] += 1
                
                if done:
                    rewards.append(episode_reward)
                    scores.append(info['final_score'])
                    break

        issues = []
        
        if all(r <= -50 for r in rewards):
            issues.append("All rewards very negative (< -50)")
        
        if len(action_counts) < len(space_config['actions']) * 0.7:
            issues.append(f"Only {len(action_counts)}/{len(space_config['actions'])} actions used")
        
        if all(s > 0.999 for s in scores):
            issues.append("All scores > 0.999 (no improvements)")
        
        if issues:
            for issue in issues:
                print(f"     - {issue}")
            return False, issues
        else:
            return True, []
            
    except Exception as e:
        return False, [f"Exception: {e}"]


def train_action_space(space_name, space_config, base_output_dir, episodes=1000):
    output_dir = os.path.join(base_output_dir, space_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create optimized environment
    env = OptimizedScoreBasedMalwareEnv(
        action_list=space_config['actions'],
        success_threshold=0.998,  
        reward_scale=1000,        
        max_turns=25              
    )

    agent = create_acer_agent(env)
    
    print(f"Starting overnight training for {episodes} episodes...")
    print(f"Estimated time: {episodes * 0.5 / 60:.1f} minutes")
    start_time = time.time()
    
    try:
        chainerrl.experiments.train_agent_with_evaluation(
            agent, env,
            steps=episodes * 25,  
            max_episode_len=25,
            eval_interval=200,    
            eval_n_runs=10,       
            outdir=output_dir
        )
        
        training_time = time.time() - start_time
        
        # Save detailed statistics
        action_stats = env.get_action_stats()
        family_stats = env.get_family_stats()
        
        # Save action effectiveness analysis
        action_df = pd.DataFrame([
            {
                'action': action,
                'times_used': stats['used'],
                'success_count': stats['success'],
                'success_rate': stats['success'] / max(stats['used'], 1),
                'avg_improvement': stats['avg_improvement']
            }
            for action, stats in action_stats.items()
        ])
        action_df.to_csv(os.path.join(output_dir, 'action_effectiveness.csv'), index=False)
        
        # Save family analysis
        family_df = pd.DataFrame([
            {
                'family': family,
                'attempts': stats['attempts'],
                'successes': stats['successes'],
                'success_rate': stats['successes'] / max(stats['attempts'], 1),
                'avg_final_score': stats['avg_score']
            }
            for family, stats in family_stats.items()
        ])
        family_df.to_csv(os.path.join(output_dir, 'family_analysis.csv'), index=False)
        
        # Save configuration
        config = {
            'space_name': space_name,
            'space_config': space_config,
            'training_episodes': episodes,
            'training_time_minutes': training_time / 60,
            'environment_config': {
                'success_threshold': 0.998,
                'reward_scale': 1000,
                'reward_clip': 100,
                'max_turns': 25
            },
            'final_stats': {
                'total_actions_used': sum(stats['used'] for stats in action_stats.values()),
                'total_successes': sum(stats['success'] for stats in action_stats.values()),
                'families_encountered': len(family_stats)
            }
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        return True, action_stats, family_stats
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, {}, {}


def create_comprehensive_analysis(base_output_dir, action_spaces):
    all_results = {}
    
    for space_name in action_spaces.keys():
        space_dir = os.path.join(base_output_dir, space_name)

        progress_file = os.path.join(space_dir, 'progress.csv')
        if os.path.exists(progress_file):
            df = pd.read_csv(progress_file)
            all_results[space_name] = {
                'progress': df,
                'action_stats': None,
                'family_stats': None
            }

            action_file = os.path.join(space_dir, 'action_effectiveness.csv')
            if os.path.exists(action_file):
                all_results[space_name]['action_stats'] = pd.read_csv(action_file)

            family_file = os.path.join(space_dir, 'family_analysis.csv')
            if os.path.exists(family_file):
                all_results[space_name]['family_stats'] = pd.read_csv(family_file)
    
    if not all_results:
        print("❌ No results found for analysis")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Action Space Training Comparison\nBuilding on Breakthrough Results: +325 → +6008 → +18,978', fontsize=16)
    
    ax = axes[0, 0]
    for space_name, data in all_results.items():
        if 'progress' in data and data['progress'] is not None:
            df = data['progress']
            ax.plot(df['episodes'], df['mean'], label=action_spaces[space_name]['name'], linewidth=2)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
    ax.axhline(y=325, color='green', linestyle='--', alpha=0.7, label='Your Breakthrough (+325)')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Progress Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    final_rewards = []
    space_names = []
    
    for space_name, data in all_results.items():
        if 'progress' in data and data['progress'] is not None:
            final_reward = data['progress']['mean'].iloc[-1]
            final_rewards.append(final_reward)
            space_names.append(action_spaces[space_name]['name'])
    
    if final_rewards:
        bars = ax.bar(range(len(final_rewards)), final_rewards, 
                     color=['blue', 'green', 'orange', 'red'][:len(final_rewards)])
        ax.set_xticks(range(len(space_names)))
        ax.set_xticklabels(space_names, rotation=45, ha='right')
        ax.set_ylabel('Final Mean Reward')
        ax.set_title('Final Performance Comparison')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, final_rewards)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_rewards)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

    ax = axes[0, 2]
    for i, (space_name, data) in enumerate(all_results.items()):
        if data['action_stats'] is not None:
            action_df = data['action_stats']
            y_pos = np.arange(len(action_df)) + i * 0.2
            ax.barh(y_pos, action_df['success_rate'], height=0.15, 
                   label=action_spaces[space_name]['name'])
    
    ax.set_xlabel('Success Rate')
    ax.set_title('Action Success Rates by Space')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    all_families = set()
    for data in all_results.values():
        if data['family_stats'] is not None:
            all_families.update(data['family_stats']['family'].tolist())
    
    family_data = defaultdict(dict)
    for space_name, data in all_results.items():
        if data['family_stats'] is not None:
            for _, row in data['family_stats'].iterrows():
                family_data[row['family']][space_name] = row['success_rate']
    
    if family_data:
        families = list(family_data.keys())
        x = np.arange(len(families))
        width = 0.8 / len(all_results)
        
        for i, space_name in enumerate(all_results.keys()):
            rates = [family_data[family].get(space_name, 0) for family in families]
            ax.bar(x + i * width, rates, width, label=action_spaces[space_name]['name'])
        
        ax.set_xlabel('Malware Family')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Malware Family')
        ax.set_xticks(x + width * (len(all_results) - 1) / 2)
        ax.set_xticklabels(families)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    space_sizes = [len(action_spaces[space]['actions']) for space in all_results.keys()]
    space_performance = final_rewards if final_rewards else [0] * len(space_sizes)
    
    ax.scatter(space_sizes, space_performance, s=100, alpha=0.7)
    for i, space_name in enumerate(all_results.keys()):
        ax.annotate(action_spaces[space_name]['name'], 
                   (space_sizes[i], space_performance[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Action Space Size')
    ax.set_ylabel('Final Mean Reward')
    ax.set_title('Action Space Size vs Performance')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    stabilities = []
    for space_name, data in all_results.items():
        if 'progress' in data and data['progress'] is not None:
            df = data['progress']
            if len(df) > 10:
                stability = df['mean'].rolling(10).std().mean()
                stabilities.append(stability)
            else:
                stabilities.append(0)
        else:
            stabilities.append(0)
    
    if stabilities:
        bars = ax.bar(range(len(stabilities)), stabilities,
                     color=['blue', 'green', 'orange', 'red'][:len(stabilities)])
        ax.set_xticks(range(len(space_names)))
        ax.set_xticklabels(space_names, rotation=45, ha='right')
        ax.set_ylabel('Training Stability (Lower = More Stable)')
        ax.set_title('Training Stability Comparison')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, 'comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

    create_summary_report(base_output_dir, all_results, action_spaces)


def create_summary_report(base_output_dir, all_results, action_spaces):    
    best_space = None
    best_reward = float('-inf')
    
    for space_name, data in all_results.items():
        if 'progress' in data and data['progress'] is not None:
            df = data['progress']
            final_reward = df['mean'].iloc[-1]
            
            if final_reward > best_reward:
                best_reward = final_reward


    report_path = os.path.join(base_output_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    np.random.seed(123)
    
    family_map = load_family_map()

    action_spaces = create_action_spaces()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"overnight_training_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    validation_results = {}
    
    for space_name, space_config in action_spaces.items():
        success, issues = validate_action_space(space_name, space_config, num_runs=10)
        validation_results[space_name] = {'success': success, 'issues': issues}

    failed_validations = [name for name, result in validation_results.items() if not result['success']]
    
    if failed_validations:
        for name in failed_validations:
            print(f"   - {action_spaces[name]['name']}")
        
        response = input("\nContinue with remaining action spaces? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

        for name in failed_validations:
            del action_spaces[name]

    training_results = {}
    overall_start_time = time.time()
    
    for i, (space_name, space_config) in enumerate(action_spaces.items(), 1):
        print(f"\n{'='*60}")
        print(f"TRAINING {i}/{len(action_spaces)}: {space_config['name'].upper()}")
        print(f"{'='*60}")
        
        space_start_time = time.time()
        
        try:
            success, action_stats, family_stats = train_action_space(
                space_name, space_config, base_output_dir, episodes=1000
            )
            
            space_time = time.time() - space_start_time
            
            training_results[space_name] = {
                'success': success,
                'action_stats': action_stats,
                'family_stats': family_stats,
                'training_time_minutes': space_time / 60
            }
            
            if success:
                print(f"{space_config['name']} completed in {space_time/60:.1f} minutes")
            else:
                print(f"{space_config['name']} failed")


            
        except Exception as e:
            training_results[space_name] = {'success': False, 'error': str(e)}
    
    try:
        create_comprehensive_analysis(base_output_dir, action_spaces)
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    successful_trainings = sum(1 for result in training_results.values() if result.get('success', False))
    
    if successful_trainings > 0:
        print(f"\n{successful_trainings} action spaces trained")
  
        print(f"\nRESULTS PREVIEW:")
        for space_name, result in training_results.items():
            if result.get('success', False) and result.get('action_stats'):
                total_actions = sum(stats['used'] for stats in result['action_stats'].values())
                total_successes = sum(stats['success'] for stats in result['action_stats'].values())
                success_rate = total_successes / max(total_actions, 1)
                print(f"   {action_spaces[space_name]['name']}: {success_rate:.1%} success rate")

if __name__ == "__main__":
    main()