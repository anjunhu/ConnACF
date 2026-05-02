"""
Subset Selection for ConnaCF

Selects a subset of users and items for faster testing and ablation studies.
Supports multiple selection strategies to ensure representative subsets.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


class SubsetSelector:
    """
    Selects representative subsets of users and items for testing.
    
    Supports multiple selection strategies:
    - most_active: Select users/items with most interactions
    - random: Random selection
    - diverse: Select diverse users/items based on interaction patterns
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.subset_config = config.get('subset_config', {})
        
        # Support both nested (subset_config.n_users) and flat (subset_n_users) formats
        # Flat format takes precedence if both are present
        self.n_users = config.get('subset_n_users', self.subset_config.get('n_users', 50))
        self.n_items = config.get('subset_n_items', self.subset_config.get('n_items', 100))
        self.strategy = config.get('subset_selection_method', self.subset_config.get('selection_strategy', 'most_active'))
        self.min_interactions = self.subset_config.get('min_interactions_per_user', 3)
        
    def select_subset(self, 
                     train_data,
                     total_users: int,
                     total_items: int) -> Tuple[Set[int], Set[int], Dict]:
        """
        Select subset of users and items based on strategy.
        
        Args:
            train_data: Training dataset
            total_users: Total number of users in dataset
            total_items: Total number of items in dataset
            
        Returns:
            Tuple of (selected_user_ids, selected_item_ids, metadata)
        """
        print(f"\n{'='*60}")
        print(f"Selecting Subset: {self.n_users} users, {self.n_items} items")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*60}\n")
        
        # Analyze interaction patterns
        user_interactions, item_interactions = self._analyze_interactions(train_data)
        
        # Select users based on strategy
        if self.strategy == 'most_active':
            selected_users = self._select_most_active_users(user_interactions)
        elif self.strategy == 'random':
            selected_users = self._select_random_users(total_users)
        elif self.strategy == 'diverse':
            selected_users = self._select_diverse_users(user_interactions, train_data)
        else:
            print(f"Unknown strategy '{self.strategy}', using 'most_active'")
            selected_users = self._select_most_active_users(user_interactions)
        
        # Select items based on selected users
        selected_items = self._select_items_for_users(
            selected_users, user_interactions, item_interactions
        )
        
        # Verify subset quality
        metadata = self._verify_subset_quality(
            selected_users, selected_items, user_interactions, item_interactions
        )
        
        print(f"\n✓ Subset selected successfully:")
        print(f"  Users: {len(selected_users)} (target: {self.n_users})")
        print(f"  Items: {len(selected_items)} (target: {self.n_items})")
        print(f"  Avg interactions per user: {metadata['avg_interactions_per_user']:.2f}")
        print(f"  Total interactions: {metadata['total_interactions']}")
        print(f"  Density: {metadata['density']:.4f}")
        
        return selected_users, selected_items, metadata
    
    def _analyze_interactions(self, train_data) -> Tuple[Dict, Dict]:
        """Analyze interaction patterns in training data"""
        user_interactions = defaultdict(set)
        item_interactions = defaultdict(set)
        
        # Try to access the underlying dataset directly first
        # RecBole DataLoader has a .dataset attribute with inter_feat
        if hasattr(train_data, 'dataset') and hasattr(train_data.dataset, 'inter_feat'):
            inter_feat = train_data.dataset.inter_feat
            uid_field = train_data.dataset.uid_field
            iid_field = train_data.dataset.iid_field
            
            users = inter_feat[uid_field].numpy()
            items = inter_feat[iid_field].numpy()
            
            for user, item in zip(users, items):
                user_interactions[int(user)].add(int(item))
                item_interactions[int(item)].add(int(user))
            
            print(f"[SUBSET] Analyzed {len(users)} interactions from dataset.inter_feat")
            return user_interactions, item_interactions
        
        # Fallback: Iterate through training data batches
        print(f"[SUBSET] Analyzing interactions by iterating through batches...")
        for batch_data in train_data:
            # Handle different batch formats
            if isinstance(batch_data, tuple):
                interaction = batch_data[0]
            else:
                interaction = batch_data
            
            # Get user and item IDs
            if hasattr(interaction, '__getitem__'):
                users = interaction['user_id']
                items = interaction['item_id']
            else:
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(users):
                users = users.numpy()
            if torch.is_tensor(items):
                items = items.numpy()
            
            # Handle scalar vs array
            if users.ndim == 0:
                users = [users.item()]
                items = [items.item()]
            
            for user, item in zip(users, items):
                user_interactions[int(user)].add(int(item))
                item_interactions[int(item)].add(int(user))
        
        return user_interactions, item_interactions
    
    def _select_most_active_users(self, user_interactions: Dict) -> Set[int]:
        """Select users with most interactions"""
        # Sort users by number of interactions
        user_counts = [(user, len(items)) for user, items in user_interactions.items()]
        user_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N users with minimum interactions
        selected = set()
        for user, count in user_counts:
            if count >= self.min_interactions:
                selected.add(user)
                if len(selected) >= self.n_users:
                    break
        
        # If not enough users, lower the threshold
        if len(selected) < self.n_users:
            print(f"⚠ Only {len(selected)} users with {self.min_interactions}+ interactions")
            print(f"  Lowering threshold to get {self.n_users} users...")
            for user, count in user_counts:
                selected.add(user)
                if len(selected) >= self.n_users:
                    break
        
        return selected
    
    def _select_random_users(self, total_users: int) -> Set[int]:
        """Select random users"""
        return set(np.random.choice(total_users, size=self.n_users, replace=False))
    
    def _select_diverse_users(self, user_interactions: Dict, train_data) -> Set[int]:
        """
        Select diverse users based on interaction patterns.
        Uses clustering-like approach to get users with different preferences.
        """
        # Start with most active users
        user_counts = [(user, len(items)) for user, items in user_interactions.items()]
        user_counts.sort(key=lambda x: x[1], reverse=True)
        
        selected = set()
        item_coverage = set()
        
        # Greedily select users that add most new items to coverage
        for user, count in user_counts:
            if count < self.min_interactions:
                continue
            
            user_items = user_interactions[user]
            new_items = user_items - item_coverage
            
            # Select if adds significant new coverage or we need more users
            if len(new_items) > 0 or len(selected) < self.n_users // 2:
                selected.add(user)
                item_coverage.update(user_items)
                
                if len(selected) >= self.n_users:
                    break
        
        # Fill remaining with most active
        if len(selected) < self.n_users:
            for user, count in user_counts:
                if user not in selected:
                    selected.add(user)
                    if len(selected) >= self.n_users:
                        break
        
        return selected
    
    def _select_items_for_users(self, 
                                selected_users: Set[int],
                                user_interactions: Dict,
                                item_interactions: Dict) -> Set[int]:
        """Select items that selected users have interacted with"""
        # Collect all items interacted by selected users
        candidate_items = set()
        for user in selected_users:
            candidate_items.update(user_interactions[user])
        
        # If we have more items than needed, select most popular
        if len(candidate_items) > self.n_items:
            item_counts = [(item, len(item_interactions[item])) 
                          for item in candidate_items]
            item_counts.sort(key=lambda x: x[1], reverse=True)
            selected_items = set([item for item, _ in item_counts[:self.n_items]])
        else:
            selected_items = candidate_items
        
        return selected_items
    
    def _verify_subset_quality(self,
                               selected_users: Set[int],
                               selected_items: Set[int],
                               user_interactions: Dict,
                               item_interactions: Dict) -> Dict:
        """Verify quality of selected subset"""
        # Count interactions in subset
        total_interactions = 0
        interactions_per_user = []
        
        for user in selected_users:
            user_items = user_interactions[user]
            subset_items = user_items & selected_items
            interactions_per_user.append(len(subset_items))
            total_interactions += len(subset_items)
        
        # Calculate density
        max_possible = len(selected_users) * len(selected_items)
        density = total_interactions / max_possible if max_possible > 0 else 0
        
        return {
            'total_interactions': total_interactions,
            'avg_interactions_per_user': np.mean(interactions_per_user) if interactions_per_user else 0,
            'min_interactions_per_user': min(interactions_per_user) if interactions_per_user else 0,
            'max_interactions_per_user': max(interactions_per_user) if interactions_per_user else 0,
            'density': density,
            'n_users': len(selected_users),
            'n_items': len(selected_items),
        }
    
    def create_subset_mapping(self,
                             selected_users: Set[int],
                             selected_items: Set[int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Create mappings between original and subset IDs.
        
        Returns:
            Tuple of (user_old_to_new, user_new_to_old, item_old_to_new, item_new_to_old)
        """
        # Create user mappings
        user_old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(selected_users))}
        user_new_to_old = {new_id: old_id for old_id, new_id in user_old_to_new.items()}
        
        # Create item mappings
        item_old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(selected_items))}
        item_new_to_old = {new_id: old_id for old_id, new_id in item_old_to_new.items()}
        
        return user_old_to_new, user_new_to_old, item_old_to_new, item_new_to_old
    
    def filter_dataset(self,
                      dataset,
                      selected_users: Set[int],
                      selected_items: Set[int],
                      user_mapping: Dict,
                      item_mapping: Dict):
        """
        Filter dataset to only include selected users and items.
        
        This is a placeholder - actual implementation depends on ConnaCF's dataset structure.
        """
        # This would need to be implemented based on ConnaCF's specific dataset format
        # For now, return the original dataset
        # In practice, you'd filter interactions and remap IDs
        print("⚠ Dataset filtering not yet implemented - using full dataset")
        print("  Subset selection will be applied at the agent level")
        return dataset


def filter_train_data_for_subset(train_data, subset_users: Set[int], subset_items: Set[int]):
    """
    Filter training dataset to only include interactions involving subset users and items.
    
    CRITICAL: This function filters the DATASET (inter_feat) AND updates the DataLoader's
    sampler to only use valid indices. This is necessary because PyTorch DataLoaders
    cache indices at creation time.
    
    Args:
        train_data: RecBole DataLoader (we modify its underlying dataset and sampler)
        subset_users: Set of user IDs in the subset
        subset_items: Set of item IDs in the subset
        
    Returns:
        The train_data with filtered dataset and updated sampler.
    """
    import torch
    from torch.utils.data import Sampler
    
    print(f"\n[SUBSET_FILTER] Filtering training data for subset...")
    print(f"[SUBSET_FILTER] Subset: {len(subset_users)} users, {len(subset_items)} items")
    
    # Convert to int sets for comparison
    subset_users_int = {int(u) for u in subset_users}
    subset_items_int = {int(i) for i in subset_items}
    
    # Get the underlying dataset
    dataset = train_data._dataset
    inter_feat = dataset.inter_feat
    
    # Get user and item columns
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    
    user_ids = inter_feat[uid_field].numpy()
    item_ids = inter_feat[iid_field].numpy()
    
    original_count = len(user_ids)
    print(f"[SUBSET_FILTER] Original interactions: {original_count}")
    
    # Create mask for interactions where BOTH user AND item are in subset
    mask = np.array([
        (int(u) in subset_users_int) and (int(i) in subset_items_int)
        for u, i in zip(user_ids, item_ids)
    ])
    
    filtered_count = mask.sum()
    print(f"[SUBSET_FILTER] Filtered interactions: {filtered_count}")
    print(f"[SUBSET_FILTER] Reduction: {(1 - filtered_count/original_count)*100:.1f}%")
    
    if filtered_count == 0:
        print(f"[SUBSET_FILTER] ⚠ WARNING: No interactions remain after filtering!")
        print(f"[SUBSET_FILTER] This likely means subset users/items don't have interactions together")
        return train_data
    
    # Get the indices that pass the filter
    valid_indices = np.where(mask)[0]
    
    # Filter the interaction features
    filtered_inter_feat = {}
    for col in inter_feat.columns:
        col_data = inter_feat[col]
        if torch.is_tensor(col_data):
            filtered_inter_feat[col] = col_data[valid_indices]
        else:
            filtered_inter_feat[col] = col_data[valid_indices]
    
    # Update the dataset's inter_feat - this is the KEY modification
    from recbole.data.interaction import Interaction
    dataset.inter_feat = Interaction(filtered_inter_feat)
    
    new_size = filtered_count
    print(f"[SUBSET_FILTER] New dataset size: {new_size}")
    
    # Store filtering info for verification
    train_data._subset_filtered = True
    train_data._subset_size = new_size
    train_data._subset_users = subset_users_int
    train_data._subset_items = subset_items_int
    
    # Store the valid indices (now 0 to new_size-1 since we filtered)
    train_data._filtered_indices = list(range(new_size))
    
    print(f"\n[SUBSET_FILTER] ✓ Dataset filtered successfully")
    print(f"[SUBSET_FILTER] Unique users: {len(set(filtered_inter_feat[uid_field].numpy()))}")
    print(f"[SUBSET_FILTER] Unique items: {len(set(filtered_inter_feat[iid_field].numpy()))}")
    print(f"[SUBSET_FILTER] New valid indices: 0 to {new_size-1}")
    print(f"[SUBSET_FILTER] ⚠ IMPORTANT: DataLoader must be RECREATED to use filtered data!")
    
    return train_data


def recreate_dataloader_for_filtered_dataset(config, dataset, train_data):
    """
    Recreate DataLoader after dataset filtering.
    
    This is necessary because PyTorch DataLoaders cache indices at creation time.
    After filtering inter_feat, we need a new DataLoader with correct indices.
    
    Args:
        config: RecBole config
        dataset: The dataset object (may or may not be the same as train_data._dataset)
        train_data: The original train_data (with filtered _dataset.inter_feat)
        
    Returns:
        New train_data, valid_data, test_data DataLoaders
    """
    print(f"\n[DATALOADER_RECREATE] Recreating DataLoader for filtered dataset...")
    print(f"[DATALOADER_RECREATE] dataset inter_feat size: {len(dataset.inter_feat)}")
    print(f"[DATALOADER_RECREATE] train_data._dataset inter_feat size: {len(train_data._dataset.inter_feat)}")
    
    # Get the new size from the filtered dataset
    new_size = len(train_data._dataset.inter_feat)
    print(f"[DATALOADER_RECREATE] New dataset size: {new_size}")
    
    # Get the DataLoader class from the original train_data
    dataloader_class = type(train_data)
    print(f"[DATALOADER_RECREATE] DataLoader class: {dataloader_class}")
    
    try:
        # Get original parameters - RecBole Config uses bracket notation
        batch_size = config['train_batch_size']
        shuffle = config['shuffle'] if 'shuffle' in config else True
        
        # Import RecBole's TrainDataLoader
        from recbole.data.dataloader import TrainDataLoader
        from recbole.sampler import Sampler
        
        # Get the original sampler
        original_sampler = train_data.sampler
        print(f"[DATALOADER_RECREATE] Original sampler type: {type(original_sampler)}")
        
        # Create a new sampler with the filtered dataset
        # The Sampler will build used_ids from train_data._dataset.inter_feat
        base_sampler = Sampler(
            phases=['train'],
            datasets=[train_data._dataset],
            distribution='uniform'
        )
        
        # CRITICAL: Call set_phase to get a sampler with used_ids as numpy array
        # instead of a dictionary keyed by phase name
        new_sampler = base_sampler.set_phase('train')
        
        print(f"[DATALOADER_RECREATE] Created new RecBole Sampler with phase='train'")
        print(f"[DATALOADER_RECREATE] Sampler used_ids type: {type(new_sampler.used_ids)}")
        print(f"[DATALOADER_RECREATE] Sampler used_ids shape: {new_sampler.used_ids.shape if hasattr(new_sampler.used_ids, 'shape') else 'N/A'}")
        
        # Create new DataLoader with the new sampler
        new_train_data = TrainDataLoader(
            config=config,
            dataset=train_data._dataset,
            sampler=new_sampler,
            shuffle=shuffle
        )
        
        # CRITICAL: Also update the _sampler attribute which is used for negative sampling
        new_train_data._sampler = new_sampler
        
        print(f"[DATALOADER_RECREATE] ✓ New DataLoader created successfully")
        print(f"[DATALOADER_RECREATE] New DataLoader dataset size: {len(new_train_data._dataset.inter_feat)}")
        print(f"[DATALOADER_RECREATE] New DataLoader _sampler: {type(new_train_data._sampler)}")
        
        # Return the new train_data and keep valid/test unchanged
        return new_train_data, None, None
        
    except Exception as e:
        print(f"[DATALOADER_RECREATE] ✗ Failed to create new DataLoader with RecBole Sampler: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach: Modify the original DataLoader's sampler in place
        print(f"[DATALOADER_RECREATE] Trying alternative approach: modifying original DataLoader...")
        
        try:
            from recbole.sampler import Sampler
            
            # Create a new RecBole sampler with the filtered dataset
            base_sampler = Sampler(
                phases=['train'],
                datasets=[train_data._dataset],
                distribution='uniform'
            )
            
            # Get the phase-specific sampler
            new_sampler = base_sampler.set_phase('train')
            
            # Update the original DataLoader's _sampler (used for negative sampling)
            train_data._sampler = new_sampler
            
            # Also update the sampler used for iteration
            from torch.utils.data import SequentialSampler
            train_data.sampler = SequentialSampler(range(new_size))
            
            print(f"[DATALOADER_RECREATE] ✓ Modified original DataLoader's sampler")
            return train_data, None, None
            
        except Exception as e2:
            print(f"[DATALOADER_RECREATE] ✗ Alternative approach also failed: {e2}")
            traceback.print_exc()
            
            # Final fallback: Return original train_data with warning
            print(f"[DATALOADER_RECREATE] ⚠ WARNING: Could not recreate DataLoader!")
            print(f"[DATALOADER_RECREATE] ⚠ Training may fail with index out of bounds errors")
            print(f"[DATALOADER_RECREATE] ⚠ Returning original train_data")
            
            return train_data, None, None
        
        return new_train_data, new_valid_data, new_test_data


def apply_subset_to_connacf(connacf_instance, subset_config: Dict, train_data=None):
    """
    Apply subset selection to an ConnaCF instance.
    
    This modifies the ConnaCF instance to only use a subset of users and items.
    """
    if not subset_config.get('subset_mode', False):
        print("Subset mode not enabled")
        return connacf_instance
    
    print("\n" + "="*60)
    print("Applying Subset Selection to ConnaCF")
    print("="*60)
    
    # Create selector
    selector = SubsetSelector(subset_config)
    
    # Use provided train_data or try to get from instance
    data_source = train_data if train_data is not None else getattr(connacf_instance, 'train_data', None)
    
    if data_source is None:
        print("⚠ No training data available for subset selection")
        print("  Using simplified subset selection based on user/item counts only")
        # Simplified selection without analyzing interactions
        n_users = subset_config.get('subset_config', {}).get('n_users', 50)
        n_items = subset_config.get('subset_config', {}).get('n_items', 100)
        
        # CRITICAL FIX: Use actual user_agents/item_agents keys if available
        # This ensures subset IDs match the actual agent IDs
        import random
        
        if hasattr(connacf_instance, 'user_agents') and connacf_instance.user_agents:
            available_users = list(connacf_instance.user_agents.keys())
            selected_users = set(random.sample(available_users, min(n_users, len(available_users))))
            print(f"  Using {len(selected_users)} users from user_agents (IDs: {list(selected_users)[:5]}...)")
        else:
            selected_users = set(random.sample(range(connacf_instance.n_users), min(n_users, connacf_instance.n_users)))
            print(f"  Using range-based user IDs: {list(selected_users)[:5]}...")
        
        if hasattr(connacf_instance, 'item_agents') and connacf_instance.item_agents:
            available_items = list(connacf_instance.item_agents.keys())
            selected_items = set(random.sample(available_items, min(n_items, len(available_items))))
            print(f"  Using {len(selected_items)} items from item_agents (IDs: {list(selected_items)[:5]}...)")
        else:
            selected_items = set(random.sample(range(connacf_instance.n_items), min(n_items, connacf_instance.n_items)))
            print(f"  Using range-based item IDs: {list(selected_items)[:5]}...")
        
        metadata = {
            'total_interactions': 0,
            'avg_interactions_per_user': 0,
            'min_interactions_per_user': 0,
            'max_interactions_per_user': 0,
            'density': 0,
            'n_users': len(selected_users),
            'n_items': len(selected_items),
        }
    else:
        # Select subset with full analysis
        selected_users, selected_items, metadata = selector.select_subset(
            data_source,
            connacf_instance.n_users,
            connacf_instance.n_items
        )
    
    # Create mappings
    user_old_to_new, user_new_to_old, item_old_to_new, item_new_to_old = \
        selector.create_subset_mapping(selected_users, selected_items)
    
    # Store subset information in instance
    connacf_instance.subset_mode = True
    connacf_instance.subset_users = selected_users
    connacf_instance.subset_items = selected_items
    connacf_instance.subset_metadata = metadata
    connacf_instance.user_mapping = user_old_to_new
    connacf_instance.item_mapping = item_old_to_new
    connacf_instance.user_mapping_reverse = user_new_to_old
    connacf_instance.item_mapping_reverse = item_new_to_old
    
    # Update n_users and n_items for attack framework
    connacf_instance.subset_n_users = len(selected_users)
    connacf_instance.subset_n_items = len(selected_items)
    
    # CRITICAL: Store subset info in config so model can access it
    # This allows conditional agent creation
    if hasattr(connacf_instance, 'config'):
        connacf_instance.config['subset_mode'] = True
        connacf_instance.config['subset_users'] = selected_users
        connacf_instance.config['subset_items'] = selected_items
    
    print(f"\n✓ Subset applied to ConnaCF instance")
    print(f"  Original: {connacf_instance.n_users} users, {connacf_instance.n_items} items")
    print(f"  Subset: {len(selected_users)} users, {len(selected_items)} items")
    print(f"  Reduction: {(1 - len(selected_users)/connacf_instance.n_users)*100:.1f}% users, "
          f"{(1 - len(selected_items)/connacf_instance.n_items)*100:.1f}% items")
    
    return connacf_instance
