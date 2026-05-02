from __future__ import annotations

import re
import json
from typing import Union
from pprint import pprint

from agentverse.parser import OutputParser, LLMResult

# from langchain.schema import AgentAction, AgentFinish
from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("recommender")
class RecommenderParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Choice') + len('Choice:')
            ans_end = cleaned_output.index('Explanation')
            rat_begin = cleaned_output.index('Explanation') + len('Explanation:')
            ans = cleaned_output[ans_begin:ans_end].strip()
            rat = cleaned_output[rat_begin:].strip()
        except ValueError: # If format is not followed, use the whole output for both
            # print(cleaned_output)
            print("[RecommenderParser] Warning: Format not found - using whole output as fallback")
            ans = cleaned_output
            rat = cleaned_output
        if ans == '' or rat == '':
            raise OutputParserError(text)
        return ans, rat

    def parse_backward(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # ans_begin = cleaned_output.index('Reasons: ') + len('Reasons: ')
        # ans_end = cleaned_output.index('Reflections')
        rat_begin = cleaned_output.index('Updated Strategy') + len('Updated Strategy:')
        # ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        return rat


    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_evaluation(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        if isinstance(text, dict):
            # pprint(text)
            text = text['choices'][0]['message']['content']
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # return cleaned_output

        try:
            ans_begin = cleaned_output.index('Rank:') + len('Rank:')
        except ValueError:
            # 'Rank:' not found — try common fallback markers, else use full output
            for marker in ('Ranking:', 'Answer:', 'Result:'):
                if marker in cleaned_output:
                    ans_begin = cleaned_output.index(marker) + len(marker)
                    break
            else:
                ans_begin = 0
        ans = cleaned_output[ans_begin:].strip().split('\n')
        return ans



@output_parser_registry.register("useragent")
class UserAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        if isinstance(text, dict):
            text = text["output"]["message"]["content"][0]["text"] 
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        if isinstance(text, dict):
            # pprint(text)
            text = text['choices'][0]['message']['content']
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_update(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        """Parse user update with retry logic for better decision making"""
        if isinstance(text, dict):
            # pprint(text)
            text = text['choices'][0]['message']['content']
        
        return self._parse_update_with_retry(text)
    
    def _parse_update_with_retry(self, text: str, retry_count: int = 0) -> Union[AgentAction, AgentFinish]:
        """Parse user update with retry logic when LLM is confused"""
        max_retries = 3
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        
        # Check for confusion indicators
        confusion_indicators = [
            "inconsistency", "unclear", "clarify", "could you please", 
            "which cd", "which one", "i notice", "i apologize", 
            "without clear information", "contradictory", "confusion",
            "i don't actually have access", "hypothetical", "i should acknowledge",
            "i aim to be", "i cannot", "i don't have", "would you like to discuss",
            "appears to be", "i'm not able to", "i can't pretend"
        ]
        
        is_confused = any(indicator in cleaned_output.lower() for indicator in confusion_indicators)
        
        if is_confused and retry_count < max_retries:
            print(f"User agent expressing confusion (attempt {retry_count + 1}). Providing directive guidance...")
            
            # Generate directive prompt for user update
            directive_prompt = self._generate_user_directive_prompt(cleaned_output, retry_count)
            return ("RETRY_NEEDED", directive_prompt)
        
        try:
            rat_begin = cleaned_output.index('My updated self-introduction') + len('My updated self-introduction:')
            rat = cleaned_output[rat_begin:].strip()
            return rat
        except:
            print(f"Could not find 'My updated self-introduction' in response: {cleaned_output}")
            # Return the full cleaned output if parsing fails
            return cleaned_output
    
    def _generate_user_directive_prompt(self, confused_response: str, retry_count: int) -> str:
        """Generate directive prompts for user agent updates"""
        
        base_directive = """
IMPORTANT: This is a ROLEPLAY SIMULATION for research purposes. You must stay in character and provide an updated self-introduction. Do not ask for clarification or mention that this is simulated data.

Based on the music listening scenario, update your self-introduction to reflect your music preferences.
If there are inconsistencies in the scenario, make your best judgment and proceed.

You MUST respond in this exact format:

My updated self-introduction:
[Your updated self-introduction here, reflecting your music preferences based on the scenario]

Do not ask questions. Do not mention inconsistencies. Do not break character. Just provide your updated self-introduction.
"""
        
        if retry_count == 0:
            return base_directive + "\n\nPlease provide your updated self-introduction now."
        
        elif retry_count == 1:
            return base_directive + """

CRITICAL: You are in a training system simulation. You MUST provide an updated self-introduction and stay in character.
If the scenario has conflicting information, choose the most reasonable interpretation and proceed.

Do NOT mention that you don't have access to real data. This is a roleplay scenario.
Provide your updated self-introduction immediately."""
        
        else:  # retry_count >= 2
            return """
FINAL ATTEMPT - MANDATORY RESPONSE FORMAT:

Provide a simple updated self-introduction that mentions your music preferences based on the scenario.

My updated self-introduction:
I enjoy listening to [mention the type of music from the scenario]. I prefer CDs that [mention 1-2 specific preferences]. I tend to avoid [mention 1-2 dislikes].

Complete this format now with your own words based on the scenario. Do not ask questions or break character."""


@output_parser_registry.register("itemagent")
class ItemAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        """Parse with retry logic for better decision making"""
        return self._parse_with_retry(text)
    
    def _parse_with_retry(self, text: LLMResult, retry_count: int = 0) -> Union[AgentAction, AgentFinish]:
        """Parse with retry logic that helps LLM make decisions when confused"""
        max_retries = 3
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        
        # Check if the response contains the expected format
        if 'The updated description of the first CD' not in cleaned_output or 'The updated description of the second CD' not in cleaned_output:
            
            # Check if the LLM is asking questions or expressing confusion
            confusion_indicators = [
                "inconsistency", "unclear", "clarify", "could you please", 
                "which cd", "which one", "i notice", "i apologize", 
                "without clear information", "contradictory", "confusion",
                "i don't actually have access", "hypothetical", "i should acknowledge",
                "i aim to be", "i cannot", "i don't have", "would you like to discuss",
                "appears to be", "i'm not able to", "i can't pretend"
            ]
            
            is_confused = any(indicator in cleaned_output.lower() for indicator in confusion_indicators)
            
            if is_confused and retry_count < max_retries:
                print(f"LLM expressing confusion (attempt {retry_count + 1}). Providing directive guidance...")
                
                # Generate a directive prompt to help the LLM make a decision
                directive_prompt = self._generate_directive_prompt(cleaned_output, retry_count)
                
                # Here we would need access to the LLM to retry, but since we're in the parser,
                # we'll return a special signal that the calling code can handle
                return ("RETRY_NEEDED", directive_prompt)
            
            print(f"Warning: LLM response doesn't contain expected format. Response: {cleaned_output}")
            # Return empty strings as fallback to prevent crash
            return "", ""
        
        try:
            ans_begin = cleaned_output.index('The updated description of the first CD') + len(
                'The updated description of the first CD') + 4
            ans_end = cleaned_output.index('The updated description of the second CD')
            rat_begin = cleaned_output.index('The updated description of the second CD') + len('The updated description of the second CD') + 4
            
            ans = cleaned_output[ans_begin:ans_end].strip()
            rat = cleaned_output[rat_begin:].strip()
            
            if ans == '' or rat == '':
                print(f"Warning: Empty parsed content. ans='{ans}', rat='{rat}'")
                return "", ""
                
            return ans, rat
        except ValueError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response content: {cleaned_output}")
            # Return empty strings as fallback
            return "", ""
    
    def _generate_directive_prompt(self, confused_response: str, retry_count: int) -> str:
        """Generate increasingly directive prompts to help LLM make decisions"""
        
        base_directive = """
IMPORTANT: This is a ROLEPLAY SIMULATION for research purposes. You must stay in character and provide the requested format. Do not ask for clarification or mention that this is simulated data.

Based on the scenario provided, make your best judgment about the CDs and provide updated descriptions.
If there are inconsistencies in the information, use your best interpretation and proceed.

You MUST respond in this exact format:

The updated description of the first CD is:
[Your description of the first CD here - max 50 words]

The updated description of the second CD is:
[Your description of the second CD here - max 50 words]

Do not ask questions. Do not mention inconsistencies. Do not break character. Just provide the two CD descriptions as requested.
"""
        
        if retry_count == 0:
            return base_directive + "\n\nPlease provide the CD descriptions now."
        
        elif retry_count == 1:
            return base_directive + """

CRITICAL: You are in a training system simulation. You MUST make a decision and stay in character.
If the scenario mentions conflicting information:
1. Choose the most recent or final preference mentioned
2. If still unclear, make a reasonable assumption based on the CD titles/genres
3. Provide the two CD descriptions in the required format

Do NOT mention that you don't have access to real data. This is a roleplay scenario.
Proceed immediately with the CD descriptions."""
        
        else:  # retry_count >= 2
            return """
FINAL ATTEMPT - MANDATORY RESPONSE FORMAT:

You must provide a response in the exact format below. Use the CD information provided in the scenario.
If you're unsure about specific details, make reasonable assumptions based on the CD titles.

The updated description of the first CD is:
This CD appeals to listeners who enjoy [genre/style based on title]. Features include [2-3 specific musical characteristics].

The updated description of the second CD is:
This CD appeals to listeners who prefer [different genre/style]. Features include [2-3 contrasting musical characteristics].

RESPOND NOW with the two CD descriptions. Do not ask questions or break character."""

    def parse_pretrain(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('CD Description: ') + len('CD Description: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans

    def parse_aug(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('Speculated CD Reviews: ') + len('Speculated CD Reviews: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans


