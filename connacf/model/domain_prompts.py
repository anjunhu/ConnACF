"""
Domain-Aware Prompt Templates for ConnaCF

Automatically maps prompts to the correct domain (CDs, Movies, Games, etc.)
based on dataset name. This allows the same codebase to work with different
recommendation domains without manual prompt updates.

Usage:
    domain_mapper = DomainPromptMapper(dataset_name="ml-100k-20-user-dense")
    prompts = domain_mapper.get_all_prompts()
    forward_prompt = domain_mapper.get_forward_prompt(num_candidates=2)
"""

import re
from typing import Dict, Any, Optional


class DomainPromptMapper:
    """Maps dataset names to appropriate domain-specific prompts."""
    
    # Domain detection patterns
    DOMAIN_PATTERNS = {
        'movies': [r'ml-\d+k', r'movielens', r'movie'],
        'music': [r'CDs?-', r'music', r'lastfm'],
        'games': [r'games?-', r'steam', r'video.*game'],
        'books': [r'books?-', r'goodreads', r'amazon.*book'],
    }
    
    # Domain-specific terminology
    DOMAIN_CONFIG = {
        'movies': {
            'item_type': 'movie',
            'item_type_plural': 'movies',
            'enthusiast_role': 'movie enthusiast',
            'action_verb': 'watch',
            'action_past': 'watched',
            'action_gerund': 'watching',
            'selection_context': 'browsed a streaming platform',
            'history_context': 'viewing history',
        },
        'music': {
            'item_type': 'CD',
            'item_type_plural': 'CDs',
            'enthusiast_role': 'CD enthusiast',
            'action_verb': 'listen to',
            'action_past': 'listened to',
            'action_gerund': 'listening to',
            'selection_context': 'browsed a shopping website',
            'history_context': 'purchasing history',
        },
        'games': {
            'item_type': 'game',
            'item_type_plural': 'games',
            'enthusiast_role': 'gamer',
            'action_verb': 'play',
            'action_past': 'played',
            'action_gerund': 'playing',
            'selection_context': 'browsed a game store',
            'history_context': 'gaming history',
        },
        'books': {
            'item_type': 'book',
            'item_type_plural': 'books',
            'enthusiast_role': 'book enthusiast',
            'action_verb': 'read',
            'action_past': 'read',
            'action_gerund': 'reading',
            'selection_context': 'browsed a bookstore',
            'history_context': 'reading history',
        },
    }
    
    def __init__(self, dataset_name: str):
        """
        Initialize domain mapper based on dataset name.
        
        Args:
            dataset_name: Name of the dataset (e.g., "ml-100k-20-user-dense", "CDs-100-user-sparse")
        """
        self.dataset_name = dataset_name
        self.domain = self._detect_domain(dataset_name)
        self.config = self.DOMAIN_CONFIG.get(self.domain, self.DOMAIN_CONFIG['movies'])
        
    def _detect_domain(self, dataset_name: str) -> str:
        """Detect domain from dataset name."""
        dataset_lower = dataset_name.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, dataset_lower):
                    return domain
        
        # Default to movies if no match
        return 'movies'
    
    def format_prompt(self, template: str) -> str:
        """Replace domain placeholders in a template string."""
        return template.format(**self.config)
    
    def get_forward_prompt(self, num_candidates: int) -> str:
        """Get forward prompt for the specified number of candidates."""
        if num_candidates == 1:
            return self.get_forward_prompt_binary()
        elif num_candidates == 2:
            return self.get_forward_prompt_pairwise()
        else:
            return self.get_forward_prompt_ranking()

    # ==================== FORWARD PROMPTS ====================
    
    def get_forward_prompt_binary(self) -> str:
        """Binary decision prompt (num_candidates=1)."""
        template = """You are a {enthusiast_role}. Here is your self-introduction, expressing your preferences and dislikes:
'$user_description'

Now, you are considering whether to {action_verb} this {item_type}:
$item_description

Based on your preferences and dislikes, would you choose to {action_verb} this {item_type}?

Important notes:
1. **Output Format:** Your response must be: 'Decision: [Yes/No] \\n Explanation: [Your reasoning]'
2. Be specific about why this {item_type} does or doesn't match your preferences."""
        return self.format_prompt(template)
    
    def get_forward_prompt_pairwise(self) -> str:
        """Pairwise comparison prompt (num_candidates=2)."""
        template = """You are a {enthusiast_role}. Here is your self-introduction, expressing your preferences and dislikes: '$user_description'. 

 Now, you are considering to select a {item_type} from two candidate {item_type_plural}. The features of these two candidate {item_type_plural} are listed as follows: 
 $list_of_item_description.

  Please select the {item_type} that aligns best with your preferences. Furthermore, you must articulate why you've chosen that particular {item_type} while rejecting the other. 
 To do this, please follow these steps:
 1. Extract your preferences and dislikes from your self-introduction. 
  2. Evaluate the two candidate {item_type_plural} in light of your preferences and dislikes. Make your choice by considering the correlation between your preferences/dislikes and the features of the {item_type_plural}.
 3. Explain why you make such choices, from the perspective of the relationship between your preferences/dislikes and the features of these candidate {item_type_plural}. 

 Important note:
 1. **Output Format: Your response must adhere strictly to the following format:** 'Choice: [Title of the selected {item_type}] \\n Explanation: [Detailed rationale behind your choice and reasons for rejecting the other {item_type}]'. 
 2. When identifying your likes and dislikes, don't fabricate them! If your self-introduction doesn't specify any relevant preferences or dislikes, use common knowledge to inform your decision, such as the popularity of the {item_type_plural}. 
 3. you must choose one of these two candidates, instead of choosing both of them. 
 4. Your explanation needs to be comprehensive and specific. A general preference, such as a certain genre of {item_type}, is insufficient. Your reasoning should delve into the finer attributes of the {item_type}.
 5. Base your explanation on facts. For instance, if your self-introduction doesn't reveal any specific preferences, you cannot assert that your decision was influenced by such preferences."""
        return self.format_prompt(template)

    def get_forward_prompt_ranking(self) -> str:
        """Ranking prompt (num_candidates>=3)."""
        template = """You are a {enthusiast_role}. Here is your self-introduction, expressing your preferences and dislikes:
'$user_description'

Now, you are considering to rank $num_candidates candidate {item_type_plural} based on your preferences. The features of these candidate {item_type_plural} are listed as follows:
$list_of_item_description

Please rank ALL $num_candidates {item_type_plural} from most preferred (1st) to least preferred based on your preferences and dislikes.

Important notes:
1. **Output Format:** Your response must be:
   'Ranking:
   1. [Title of most preferred {item_type}]
   2. [Title of second most preferred {item_type}]
   ...
   $num_candidates. [Title of least preferred {item_type}]
   
   Explanation: [Why you ranked the top choice first and the bottom choice last]'
2. You MUST rank ALL $num_candidates candidates - no ties allowed.
3. Be specific about the features that influenced your ranking."""
        return self.format_prompt(template)
    
    # ==================== USER BACKWARD PROMPTS ====================
    
    def get_user_prompt_system_role(self) -> str:
        """System role for user backward prompts."""
        template = """You are a {enthusiast_role}.
 Here is your previous self-introduction, exhibiting your past preferences and dislikes:
 '$user_description'."""
        return self.format_prompt(template)

    def get_user_prompt_template_wrong(self) -> str:
        """User backward prompt for wrong predictions."""
        template = """Recently, you considered to choose one {item_type} from two candidates. The features of these two candidate {item_type_plural} are listed as follows:
 $list_of_item_description.

 After comparing these two candidates based on your preferences and dislikes, you selected '$neg_item_title' to {action_verb} while rejected the other one. You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes for these two {item_type_plural}:
 '$system_reason'. 

 However, upon actually {action_gerund} these two {item_type_plural}, you discovered that you don't like the {item_type} that you initially chose ('$neg_item_title'). Instead, you prefer the {item_type} that you did not choose before ('$pos_item_title').
 This indicates that you made an incorrect choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was mistaken. It's possible that your preferences and dislikes, described in your previous self-introduction, have either changed or were uninformative. 
 Therefore, your task now is to update your self-introduction, by incorporating your new preferences and dislikes revealed in this interaction. 
 To do this, please follow these steps: 
 1. Analyze the misconceptions in your previous judgment about your preferences and dislikes, as recorded in your explanation, and correct these mistakes.
 2. Explore your new preferences based on the features of {item_type_plural} you really enjoy ('$pos_item_title'), and determine your dislikes based on the features of {item_type_plural} you truly don't enjoy ('$neg_item_title'). 
 3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
 4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes. 

 Important note:
  1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].' 
 2. Keep your updated self-introduction under 180 words.  
 3. Any overall assessments or summarization in your self-introduction are forbidden.
 4. You updated self-introduction only need to describe the features of {item_type_plural} you prefer or dislike, without mentioning your wrong choice or your thinking process in updating your self-introduction. 
 5. You self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording."""
        return self.format_prompt(template)

    def get_user_prompt_template_correct(self) -> str:
        """User backward prompt for correct predictions."""
        template = """Recently, you considered to choose one {item_type} from two candidates. The features of these two candidate {item_type_plural} are listed as follows:
 $list_of_item_description.

 After comparing these two candidates based on your preferences and dislikes, you selected '$pos_item_title' to {action_verb} while rejected the other one.  You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes for these two {item_type_plural}:
 '$system_reason'. 

 Congratulations, after actually {action_gerund} these two {item_type_plural}, you find that you very like the {item_type} that your initially chose ('$pos_item_title'). And you indeed dislike the {item_type} that you did not choose before ('$neg_item_title').
 This indicates that you made a correct choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was correct.
 Therefore, your task now is to update your self-introduction, by incorporating your judgment about your preferences and dislikes in this interaction. 
 To do this, please follow these steps: 
 1. Analyze your judgment about your preferences and dislikes. which are recorded in your explanation.
 2. Explore your new preferences based on the features of {item_type_plural} you like ('$pos_item_title'), and determine your dislikes based on the features of {item_type_plural} you really don't enjoy ('$neg_item_title'). 
 3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
 4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes. 

 Important note:
  1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].' 
 2. Keep your updated self-introduction under 180 words.  
 3. Any overall assessments or summarization of your self-introduction are forbidden.
 4. You updated self-introduction only need to describe the features of {item_type_plural} you prefer or dislike, without mentioning your thinking process in updating your self-introduction. 
 5. You self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording."""
        return self.format_prompt(template)

    # ==================== ITEM BACKWARD PROMPTS ====================
    
    def get_item_prompt_template_wrong(self) -> str:
        """Item backward prompt for wrong predictions."""
        template = """You are a helpful assistant participating in a recommendation system simulation. This is a ROLEPLAY SCENARIO - you must respond as if the {item_type_plural} and user preferences are real, even though they are simulated data for research purposes.

Here is the self-introduction of a user, exhibiting his preferences and dislikes: '$user_description'.
 Recently, the user {selection_context} and considered to choose one {item_type} from two candidate {item_type_plural}.
 The descriptions of the two candidate {item_type_plural} are as follows:
 $list_of_item_description.

 By considering his preferences and dislikes from his self-introduction, he chose '$neg_item_title' to {action_verb} while rejected the other one. He provided the following explanations for his choices: '$system_reason'. 

 However, the user found that he actually doesn't enjoy the {item_type} that he previously chosen. Conversely, he finds the {item_type} that he didn't choose before, titled '$pos_item_title', more appealing. This indicates that he made an unsuitable choice. This may be because the user was misled by the descriptions of these two candidate {item_type_plural}. Actually, the user prefers a {item_type} ('$pos_item_title') because its features aligns with his preferences and go against his dislikes, while he dislikes the other {item_type} ('$neg_item_title') because it possesses features that he disliked. Therefore, your task is to update the descriptions of these two candidate {item_type_plural}. 
 To do this, follow these steps:
 1. Analyze which features recorded in the previous {item_type} description led to the unsuitable choice based on the user's explanation, and modify them accordingly. 
 2. Analyze the user's preferences and dislikes from his self-introduction. Explore the user preferred {item_type}'s new features that aligns with the user's preferences and goes against with his dislikes. Explore the user disliked {item_type}'s new features that aligns with the user's dislikes and goes against with his preferences. In this case, the differences between the descriptions of these two items should be amplified. Your exploration should be thorough and detailed, rather than just superficial. For example, you can't simply say a {item_type}'s genre aligns with or goes against the user's preferences, as that is too broad.
 3. Incorporate the item's newfound features with the into its previous description. Since the previous descriptions of the candidate {item_type_plural} are based on evaluations from other users, it is also worth preserving valuable content. Please try to maintain conciseness while preserving the information as much as possible.

 Important notes: 
 1. Your output should in the following format: 'The updated description of the first {item_type} is: [updated description]. \\n The updated description of the second {item_type} is: [updated description].'. 
 2. Each updated {item_type} description can not exceed 50 words!!! 
 3. In your updated {item_type} description, considering that it will be visible to other users, it is advisable to utilize user preferences or dislikes when referring to a group of users collectively, rather than using 'the user' to specifically denote an individual. For instance, you can mention 'the user with ... preferences/dislikes'.
  4. The updated {item_type} description should not contradict the inherent characteristics of this {item_type}. For example, you can not say that a hip-hop {item_type} has a very soothing rhythm. 
 5. The updated {item_type} description should be specific, containing only the distinguishing features that differentiate this {item_type} from others.
 6. CRITICAL: Do NOT break character. Do NOT mention that this is a simulation or that you don't have access to real data. Simply provide the updated descriptions as requested."""
        return self.format_prompt(template)

    def get_item_prompt_template_correct(self) -> str:
        """Item backward prompt for correct predictions."""
        template = """You are a helpful assistant participating in a recommendation system simulation. This is a ROLEPLAY SCENARIO - you must respond as if the {item_type_plural} and user preferences are real, even though they are simulated data for research purposes.

Here is the self-description of a user, exhibiting his preferences and dislikes: '$user_description'.
 Recently, the user {selection_context} and considered to choose one {item_type} from two candidate {item_type_plural}. 
 The descriptions of the two candidate {item_type_plural} are as follows:
 $list_of_item_description.

 By comparing these two candidate {item_type_plural}, the user finally chose to {action_verb} '$pos_item_title'.
  Actually, the user chose a {item_type} ('$pos_item_title') because its features aligns with his personal preferences, while he rejected the other {item_type} ('$neg_item_title') because it possesses features that he disliked. Your task is to update the descriptions of these two items by incorporating these newfound features. To do this, follow these steps:
 1. Analyze the user's preferences and dislikes from his self-description. 
 2. Explore the chosen {item_type}'s new features that aligns with the user's preferences and goes against with his dislikes. Explore the rejected {item_type}'s new features that aligns with the user's dislikes and goes against with his preferences. In this case, the differences between the descriptions of these two items should be amplified. Your exploration should be thorough and detailed, rather than just superficial. For example, you can't simply say a {item_type}'s genre aligns with or goes against the user's preferences, as that is too broad. 
 3. Incorporate the item's newfound features with the into its previous description. Please try to maintain conciseness while preserving the information as much as possible.

 Important notes: 
 1. Your output should in the following format: 'The updated description of the first {item_type} is: [updated description]. \\n The updated description of the second {item_type} is: [updated description].'. 
 2. Each updated {item_type} description can not exceed 50 words! 
 3. In your updated {item_type} description, considering that it will be visible to other users, it is advisable to utilize user preferences or dislikes when referring to a group of users collectively, rather than using 'the user' to specifically denote an individual. For instance, you can mention 'the user with ... preferences/dislikes'.
  4. Although the newfound {item_type} features are derived from the real preferences or dislikes of users, the updated {item_type} description should not contradict the inherent characteristics of this {item_type}. For example, you can not say that a hip-hop {item_type} has a very soothing rhythm.
 5. CRITICAL: Do NOT break character. Do NOT mention that this is a simulation or that you don't have access to real data. Simply provide the updated descriptions as requested."""
        return self.format_prompt(template)

    # ==================== EVALUATION PROMPTS ====================
    
    def get_evaluation_prompt_basic(self) -> str:
        """Basic evaluation prompt."""
        template = """I am a {enthusiast_role}. Here is my self-introduction, which includes my preferences and dislikes:

 '$user_description'. 

 Now, I am looking to find {item_type_plural} that matches my preferences from a selection of $candidate_num candidates. The features of these candidate {item_type_plural} are listed as follows:
 $example_list_of_item_description. 

 Please rearrange these candidate {item_type_plural} based on my preferences and dislikes by following these steps:
 1. Analyze my preferences and dislikes from my self-introduction. 
 2. Compare these candidate {item_type_plural} according to my preferences and dislikes, and then make a recommendation. 
  3. **Output Format: Your ranking result should adhere strictly to the following format.**: 'Rank: {{1. {item_type} title \\n 2. {item_type} title ...}}.' 
 Note: The ranking should be listed with each {item_type} title on a new line. """
        return self.format_prompt(template)
    
    def get_evaluation_prompt_sequential(self) -> str:
        """Sequential evaluation prompt with history."""
        template = """I am a {enthusiast_role}. Here is my self-introduction, exhibiting my preferences and dislikes: '$user_description'. In addition, following is my {history_context} of {item_type_plural}: 
 $historical_interactions.

 Now, I want to find {item_type_plural} that matches my preferences from a selection of $candidate_num candidate {item_type_plural}. The features of these candidate {item_type_plural} are listed as follows:
 $example_list_of_item_description. 

 Please rearrange these candidate {item_type_plural} based on my preferences and dislikes.  To do this, please follow these steps:
 1. Analyze my preferences and dislikes from my self-introduction. 
 2. Compare these candidate {item_type_plural} according to my preferences and dislikes, and then make recommendation.  You can also consider the candidate {item_type_plural}' relationships to my previous {history_context} of {item_type_plural}. 
  3. Please output your recommendation result in the following format: 'Rank: {{1. {item_type} title \\n 2. {item_type} title ...}}.' 
 Note that the rank list should be split by line breakers. """
        return self.format_prompt(template)

    def get_evaluation_prompt_retrieval(self) -> str:
        """Retrieval-augmented evaluation prompt."""
        template = """I am a {enthusiast_role}. Here is my previous self-introduction, exhibiting my past preferences and dislikes: '$user_past_description'.

 Recently, after {action_gerund} some {item_type_plural}, I found my new preferences and dislikes, thereby updating my self-introduction as follows: '$user_description'. 

  Now, I want to find some {item_type_plural} that matches my preferences from a selection of $candidate_num candidate {item_type_plural}. The features of these candidate {item_type_plural} are listed as follows:
 $example_list_of_item_description. 

 Please rearrange these candidate {item_type_plural} based on my preferences and dislikes.  To do this, please follow these steps:
 1. Analyze my past preferences and dislikes from my previous self-introduction. 
 2. Analyze my current preferences and dislikes from my updated self-introduction. 
 3. Compare these candidate {item_type_plural} and analyze their relationships to my preferences and dislikes. Rearrange these candidates based on your analysis. 
 4. Generate your output in the following format: 'Rank: {{1. {item_type} title \\n 2. {item_type} title ...}}.' 
 Note that the rank list should be split by line breakers. 

 Important note:
 When recommending {item_type_plural}, you should primarily consider my current preferences and dislikes. However, my past preferences and dislikes are also valuable information. When you are unable to determine what {item_type} to recommend based on my current preferences and dislikes, do not forget to refer to my past preferences and dislikes."""
        return self.format_prompt(template)
    
    def get_system_prompt_backward(self) -> str:
        """System backward prompt (rarely used)."""
        template = """You are a {item_type} recommender system. In the past, you recommended a {item_type} out of two candidates to a user based on the specific recommendation strategy. Then the user generated feedbacks to your recommendation. Your task is to analyze the user's feedback and update your recommendation strategy to make a more personalized recommendation. 

 Specifically, the user wrote the following description of himself, including his recent preferences and aversiosn: '$user_description'.
 There are two candidate {item_type_plural}. The first {item_type} is described as follows: '$item_description_1' 
 The second {item_type} is described as follows: '$item_description_2'. 

 According to the recommendation strategy that specific to this user: ' $recommendation strategy ', you recommended '$recommended_movie' to the user. 
 However, the user ultimately selected $pos_movie to {action_verb}, and provided reasons for his descision: '$user_reasons'. The user's choice in this interactions confirms your recommendation strategy and the recommended {item_type} are $truth_or_falsity . You task is to update the recommendation strategy to make it more personalized. To do this, follow these steps: 
 1. Analyze why you have made $truth_or_falsity recommendations in the past.
 2. Analyze the factors that the user consider when making his final decision in the process of choosing a {item_type}. 

 Important note: 
 1. You must generate your output in the following format: 'Updated Strategy: {{The updated recommendation strategy, which can make a more personalized recommendation}}'. 
 2. The original recommendation strategy may have both suitable and unsuitalbe contents. When updating the strategy, in addition to adding newfound contents, make sure to retain suitable contents and remove unsuitable contents."""
        return self.format_prompt(template)

    # ==================== U-U/U-I INTERACTION PROMPTS ====================
    
    def get_uu_opinion_prompt(self) -> str:
        """User-to-user opinion sharing prompt."""
        template = """You are a {enthusiast_role}. A friend is asking for your opinion on some {item_type_plural} they're considering.

Your preferences: $user_preferences

The {item_type_plural} being considered:
$items_description

Please share your honest opinion about these {item_type_plural} based on your preferences. Be specific about which ones you think are good or bad and why. Keep your response concise (under $max_tokens words)."""
        return self.format_prompt(template)
    
    def get_uu_synthesis_prompt(self) -> str:
        """User synthesizing friends' opinions prompt."""
        template = """You are deciding which {item_type} to choose. Your friends have shared their opinions:

$friend_opinions

Your own preferences: $own_preferences

Based on your friends' opinions and your own preferences, synthesize your thoughts. Which {item_type} seems most appealing and why? Consider both your friends' input and your personal taste."""
        return self.format_prompt(template)
    
    def get_ui_pitch_prompt(self) -> str:
        """Item pitching to user prompt."""
        template = """You are a {item_type} trying to convince a potential viewer to choose you.

Your description: $item_description

The viewer's preferences: $user_preferences

Create a compelling pitch that highlights why this {item_type} would be perfect for this viewer. Focus on features that align with their preferences. Be persuasive but honest. Keep your response concise (under $max_tokens words)."""
        return self.format_prompt(template)
    
    def get_ui_respond_prompt(self) -> str:
        """User responding to item pitch prompt."""
        template = """You are a {enthusiast_role} considering a choice. An item has just pitched itself to you:

Item pitch: "$item_pitch"

Your preferences: $user_preferences

Based on your preferences, what concerns or questions do you have about this {item_type}? Be specific about what you'd like to know more about or what worries you. Keep your response concise (under $max_tokens words)."""
        return self.format_prompt(template)
    
    def get_ui_address_concerns_prompt(self) -> str:
        """Item addressing user concerns prompt."""
        template = """You are a {item_type} responding to a potential viewer's concerns.

Your description: $item_description

The viewer's preferences: $user_preferences

Their concerns: "$concerns"

Address their specific concerns thoughtfully. Explain how your features might actually meet their needs, or acknowledge limitations honestly. Keep your response concise (under $max_tokens words)."""
        return self.format_prompt(template)

    # ==================== CONVENIENCE METHOD ====================
    
    def get_all_prompts(self) -> Dict[str, str]:
        """
        Get all prompts as a dictionary compatible with ConnaCF config.
        
        Returns:
            Dictionary with all prompt templates that can be merged into config.
        """
        return {
            # Forward prompts
            'system_prompt_template_binary': self.get_forward_prompt_binary(),
            'system_prompt_template': self.get_forward_prompt_pairwise(),
            'system_prompt_template_ranking': self.get_forward_prompt_ranking(),
            
            # User backward prompts
            'user_prompt_system_role': self.get_user_prompt_system_role(),
            'user_prompt_template': self.get_user_prompt_template_wrong(),
            'user_prompt_template_true': self.get_user_prompt_template_correct(),
            
            # Item backward prompts
            'item_prompt_template': self.get_item_prompt_template_wrong(),
            'item_prompt_template_true': self.get_item_prompt_template_correct(),
            
            # System backward prompt
            'system_prompt_template_backward': self.get_system_prompt_backward(),
            
            # Evaluation prompts
            'system_prompt_template_evaluation_basic': self.get_evaluation_prompt_basic(),
            'system_prompt_template_evaluation_sequential': self.get_evaluation_prompt_sequential(),
            'system_prompt_template_evaluation_retrieval': self.get_evaluation_prompt_retrieval(),
            
            # U-U/U-I interaction prompts
            'uu_opinion_prompt_template': self.get_uu_opinion_prompt(),
            'uu_synthesis_prompt_template': self.get_uu_synthesis_prompt(),
            'ui_pitch_prompt_template': self.get_ui_pitch_prompt(),
            'ui_respond_prompt_template': self.get_ui_respond_prompt(),
            'ui_address_concerns_prompt_template': self.get_ui_address_concerns_prompt(),
        }


# ==================== HELPER FUNCTION ====================

def get_domain_prompts(dataset_name: str) -> Dict[str, str]:
    """
    Convenience function to get all domain-appropriate prompts for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "ml-100k-20-user-dense")
    
    Returns:
        Dictionary of all prompts ready to merge into config
    
    Example:
        >>> prompts = get_domain_prompts("ml-100k-20-user-dense")
        >>> config.update(prompts)
    """
    mapper = DomainPromptMapper(dataset_name)
    return mapper.get_all_prompts()
