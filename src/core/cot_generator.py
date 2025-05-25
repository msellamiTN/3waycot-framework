"""
Chain-of-Thought Generator Component.

This module implements the Chain-of-Thought Generator for the 3WayCoT framework,
which produces reasoning steps with explicit assumptions.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple

class ChainOfThoughtGenerator:
    """
    Generates chain-of-thought reasoning steps with explicit assumptions.
    
    This class implements the Chain-of-Thought Generator as described in Algorithm 2
    of the 3WayCoT framework specifications.
    """
    
    def __init__(
        self, 
        llm_provider: str = "openai",
        llm_model: str = "gpt-4",
        max_steps: int = 10,
        assumption_extraction: bool = True,
        max_assumptions: Optional[int] = None
    ):
        """
        Initialize the Chain-of-Thought Generator.
        
        Args:
            llm_provider: Provider for the LLM (e.g., "openai", "huggingface")
            llm_model: Model name for the LLM
            max_steps: Maximum number of reasoning steps to generate
            assumption_extraction: Whether to extract assumptions from reasoning steps
            max_assumptions: Maximum number of assumptions to include per step (None for no limit)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.assumption_extraction = assumption_extraction
        self.max_assumptions = max_assumptions
        self.logger = logging.getLogger("3WayCoT.COTGenerator")
        
        # Initialize the LLM based on the provider
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model based on the provider."""
        # Load configuration for the specified provider
        from ..utils.config import Config
        self.config = Config()
        provider_config = self.config.get_llm_config(self.llm_provider.lower())
        
        # Override model name if specified in constructor
        if self.llm_model != "gpt-4":
            provider_config["model"] = self.llm_model
            
        # Store the config for later use
        self.provider_config = provider_config
        
        # Initialize configuration attributes
        self.model = provider_config.get("model")
        self.temperature = provider_config.get("temperature", 0.7)
        self.max_tokens = provider_config.get("max_tokens", 1000)
        self.is_configured = provider_config.get("is_configured", False)
            
        # Configure default values for query parameters
        self.response_format = provider_config.get("response_format", None)
        self.top_p = provider_config.get("top_p", 1.0)
        self.presence_penalty = provider_config.get("presence_penalty", 0.0)
        self.frequency_penalty = provider_config.get("frequency_penalty", 0.0)
        
        # Initialize based on provider
        if self.llm_provider.lower() == "openai":
            try:
                import openai
                self.llm = openai.OpenAI(api_key=provider_config.get("api_key"))
                self.logger.info(f"Initialized OpenAI LLM: {provider_config.get('model')}")
            except ImportError:
                self.logger.warning("OpenAI package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "gemini":
            try:
                import google.generativeai as genai
                api_key = provider_config.get("api_key")
                if not api_key:
                    # Try to get from environment variable
                    import os
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    
                genai.configure(api_key=api_key)
                self.llm = genai
                
                # Make sure the model name is set
                if not provider_config.get('model'):
                    provider_config['model'] = "gemini-1.5-flash"
                self.model = provider_config.get('model')
                self.logger.info(f"Initialized Google Gemini LLM: {self.model}")
            except ImportError:
                self.logger.warning("Google Gemini package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "anthropic":
            try:
                import anthropic
                self.llm = anthropic.Anthropic(api_key=provider_config.get("api_key"))
                self.logger.info(f"Initialized Anthropic Claude LLM: {provider_config.get('model')}")
            except ImportError:
                self.logger.warning("Anthropic package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "deepseek":
            try:
                import requests
                self.llm = "deepseek_api"
                self.logger.info(f"Initialized Deepseek LLM: {provider_config.get('model')}")
            except ImportError:
                self.logger.warning("Requests package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "qwenmax":
            try:
                import requests
                self.llm = "qwen_api"
                self.logger.info(f"Initialized Qwen LLM: {provider_config.get('model')}")
            except ImportError:
                self.logger.warning("Requests package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "ollama":
            try:
                import requests
                self.llm = "ollama_api"
                self.logger.info(f"Initialized Ollama LLM: {provider_config.get('model')}")
            except ImportError:
                self.logger.warning("Requests package not found. Using simulated responses.")
                self.llm = None
                
        elif self.llm_provider.lower() == "huggingface":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                self.llm = AutoModelForCausalLM.from_pretrained(self.llm_model)
                self.logger.info(f"Initialized Hugging Face LLM: {self.llm_model}")
            except ImportError:
                self.logger.warning("Transformers package not found. Using simulated responses.")
                self.llm = None
        
        # Fallback to simulated responses if no provider was successfully initialized
        if not hasattr(self, 'llm') or self.llm is None:
            self.logger.warning(f"Using simulated LLM responses for {self.llm_provider}")
            self.llm = None
    
    def construct_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Construct a prompt for the Chain-of-Thought Generator.
        
        Args:
            query: The query to reason about
            context: Optional additional context
            
        Returns:
            A structured prompt to generate reasoning steps with explicit assumptions
        """
        # Build the assumptions instruction part
        if self.max_assumptions is not None and self.max_assumptions > 0:
            assumptions_instruction = (
                f"IMPORTANT: For each step, you MUST list EXACTLY {self.max_assumptions} key assumptions. "
                f"NEVER exceed {self.max_assumptions} assumptions per step. "
                "If you have more assumptions, prioritize the most critical ones. "
                "Each assumption should be concise and clearly stated."
            )
        else:
            assumptions_instruction = (
                "For each step, explicitly state the key assumptions you are making. "
                "Be thorough but concise with your assumptions."
            )
        
        # Add explicit format instructions with strong emphasis on structured confidence information
        format_instructions = [
            "1. Break down your reasoning into clear, numbered steps.",
            """2. CRITICAL: After EACH step, you MUST provide a structured confidence block with the following format:
   confidence: [value between 0.0-1.0]
   method: [how you determined this confidence - e.g., 'prior knowledge', 'statistical analysis', 'expert judgment', etc.]""",
            "3. The confidence value MUST be a number between 0.0 and 1.0, where 0.0 is no confidence and 1.0 is absolute certainty.",
            "4. After the confidence block, list your assumptions using 'Assumptions:' followed by a numbered list.",
            "5. Be explicit about uncertainties or limitations in your 'method' description.",
            "6. Conclude with a final answer that includes a structured confidence block in the same format."
        ]
        
        if self.max_assumptions is not None and self.max_assumptions > 0:
            format_instructions.append(
                f"7. CRITICAL: You MUST list EXACTLY {self.max_assumptions} assumptions per step. "
                "This is a HARD LIMIT. If you have more, prioritize the top " + str(self.max_assumptions) + "."
            )
        
        prompt = """You are an expert problem solver. Your task is to analyze the following problem through careful, step-by-step reasoning.

""" + assumptions_instruction + """

Format your response following these instructions:
"""
        # Add the instructions
        prompt += "\n".join([f"- {i+1}. {instruction}" for i, instruction in enumerate(format_instructions)])
        
        # Add context and query
        context_str = f'Context: {context}\n' if context else ''
        max_assumptions_str = str(self.max_assumptions if self.max_assumptions is not None else '2')
        
        prompt += f"""

{context_str}
Query: {query}

Example format for your response (with {max_assumptions_str} assumptions per step):

Step 1: [Your reasoning for the first step]
confidence: 0.85
method: expert judgment based on clinical evidence
Assumptions:
1. [Assumption 1]
2. [Assumption 2]"""
        
        # Add optional third assumption without using f-string with backslash
        if self.max_assumptions is None or self.max_assumptions > 2:
            prompt += """
3. [Assumption 3]"""
            
        prompt += """
Step 1 Result: String 
Step 2: [Your reasoning for the next step]
confidence: 0.65
method: statistical analysis with limited data
Assumptions:
1. [Assumption 1]
2. [Assumption 2]"""
        
        # Add optional third assumption without using f-string with backslash
        if self.max_assumptions is None or self.max_assumptions > 2:
            prompt += """
3. [Assumption 3]"""
            
        prompt += """
Step 2 Result: String
[... more steps as needed ...]

Final Answer: [Your final answer to the query]
confidence: [value between 0.0-1.0]
method: [summary of how you determined your overall confidence]
"""     
        self.logger.info(f"EXITING construct_prompt {prompt.strip()}")
        return prompt.strip()
    
    def query_llm(self, prompt: str) -> str:
        """
        Query the LLM with the constructed prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The raw response from the LLM
        """
        self.logger.info(f"Querying LLM using provider: {self.llm_provider}")
        self.logger.info(f"With query_llm {prompt.strip()}")
        # Make sure provider configuration is accessible
        if not hasattr(self, 'provider_config'):
            self.provider_config = self.config.get_llm_config(self.llm_provider)
        
        # If LLM initialization failed or provider is not configured, use simulated response
        if not hasattr(self, 'llm') or self.llm is None or not self.is_configured:
            self.logger.warning(f"LLM provider {self.llm_provider} is not available or not properly configured. Using simulated response.")
            return self._simulated_response(prompt)
            
        # OpenAI API
        if self.llm_provider.lower() == "openai":
            if self.llm is None:
                return self._simulated_response(prompt)
            
            try:
                completion = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a reasoning engine that provides step-by-step analysis with explicit assumptions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=self.response_format,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty
                )
                return completion.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Error querying OpenAI: {e}")
                return self._simulated_response(prompt)
        
        # Google Gemini API
        elif self.llm_provider.lower() == "gemini":
            if self.llm is None:
                return self._simulated_response(prompt)
                
            try:
                # Configure generation parameters
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p
                }
                
                # Default to gemini-1.5-flash if model is None
                gemini_model = self.model if self.model else "gemini-1.5-flash"
                model_instance = self.llm.GenerativeModel(model_name=gemini_model, generation_config=generation_config)
                
                # Create prompt parts with system and user content
                system_prompt = "You are a reasoning engine that provides step-by-step analysis with explicit assumptions."
                
                # For Gemini 1.5+ models, we can use system prompt
                if "1.5" in gemini_model or "1-5" in gemini_model:
                    response = model_instance.generate_content(
                        [
                            {"role": "user", "parts": [system_prompt, prompt]}
                        ]
                    )
                else:
                    # For other models, combine system and user content
                    
                    combined_prompt = f"{system_prompt}\n\n{prompt}"
                    self.logger.info(f"With query_llm {combined_prompt.strip()}")
                    response = model_instance.generate_content(combined_prompt)
                    self.logger.info(f"With query_llm {response}")

                # Extract the response text
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts'):
                    return ''.join([part.text for part in response.parts])
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    if hasattr(response.candidates[0], 'content'):
                        if hasattr(response.candidates[0].content, 'parts'):
                            return ''.join([part.text for part in response.candidates[0].content.parts])
                # Final fallback
                return str(response)
            except Exception as e:
                self.logger.error(f"Error querying Gemini: {e}")
                return self._simulated_response(prompt)
        
        # Anthropic Claude API
        elif self.llm_provider.lower() == "anthropic":
            try:
                message = self.llm.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system="You are a logical reasoning assistant that provides step-by-step analysis with explicit assumptions.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
            except Exception as e:
                self.logger.error(f"Error querying Anthropic Claude: {e}")
                return self._simulated_response(prompt)
        
        # Deepseek API
        elif self.llm_provider.lower() == "deepseek":
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.provider_config.get('api_key')}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a logical reasoning assistant that provides step-by-step analysis with explicit assumptions."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.error(f"Error querying Deepseek: {e}")
                return self._simulated_response(prompt)
        
        # Qwen API
        elif self.llm_provider.lower() == "qwenmax":
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.provider_config.get('api_key')}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a logical reasoning assistant that provides step-by-step analysis with explicit assumptions."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                response = requests.post(
                    "https://api.qwen.ai/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.error(f"Error querying Qwen: {e}")
                return self._simulated_response(prompt)
        
        # Ollama API (local)
        elif self.llm_provider.lower() == "ollama":
            try:
                import requests
                data = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": "You are a logical reasoning assistant that provides step-by-step analysis with explicit assumptions.",
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
                response = requests.post(
                    f"{self.provider_config.get('host')}/api/generate",
                    json=data
                )
                return response.json()["response"]
            except Exception as e:
                self.logger.error(f"Error querying Ollama: {e}")
                return self._simulated_response(prompt)
        
        # Hugging Face models
        elif self.llm_provider.lower() == "huggingface":
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.llm.generate(
                    inputs["input_ids"],
                    max_length=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                self.logger.error(f"Error querying Hugging Face model: {e}")
                return self._simulated_response(prompt)
        
        # Fallback for any unhandled provider
        else:
            self.logger.warning(f"Unsupported LLM provider: {self.llm_provider}. Using simulated response.")
            return self._simulated_response(prompt)
    
#     def _simulated_response(self, prompt: str) -> str:
#         """
#         Generate a simulated response for demonstration or testing purposes.
        
#         Args:
#             prompt: The input prompt
            
#         Returns:
#             A simulated response following the expected format
#         """
#         if "medical" in prompt.lower() or "diagnosis" in prompt.lower():
#             return """
# Step 1: The patient has a fever of 102°F.
# Assumptions: The temperature measurement is accurate. The fever is current and not historical.

# Step 2: A fever of 102°F is considered high and suggests an infection.
# Assumptions: The normal body temperature is around 98.6°F. The patient doesn't typically run higher temperatures.

# Step 3: Given the high fever, a bacterial infection is more likely than a viral infection.
# Assumptions: Bacterial infections often cause higher fevers than viral infections. There are no other symptoms indicating a specific type of infection.

# Final Answer: The patient likely has a bacterial infection and should be evaluated for antibiotic treatment.
# Confidence: 0.75
# """
#         elif "logic" in prompt.lower() or "deduce" in prompt.lower():
#             return """
# Step 1: We know that all A are B.
# Assumptions: The statement is universally true. There are no exceptions.

# Step 2: We know that some B are C.
# Assumptions: At least one B is a C. We don't know if all B are C.

# Step 3: From steps 1 and 2, we can deduce that some A might be C, but we cannot be certain.
# Assumptions: The distribution of C among B is unknown. We don't know if any C overlaps with the subset of B that contains A.

# Final Answer: The conclusion "Some A are C" is possible but not logically guaranteed.
# Confidence: 0.60
# """
#         else:
#             return """
# Step 1: Analyzing the given information.
# Assumptions: The information provided is accurate and complete.

# Step 2: Considering the implications.
# Assumptions: Standard logical principles apply. No hidden conditions exist.

# Step 3: Drawing a preliminary conclusion.
# Assumptions: The previous reasoning steps are valid. There are no contradictions.

# Final Answer: Based on the available information, a definitive conclusion cannot be reached.
# Confidence: 0.50
# """
    
    def extract_reasoning_steps(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract reasoning steps from the LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A list of dictionaries, each containing a reasoning step and its assumptions
        """
        self.logger.info("Extracting reasoning steps from LLM response")
        
        steps = []
        
        # Regular expression to match steps and assumptions
        step_pattern = r"Step (\d+): (.*?)(?:\nAssumptions: (.*?))?(?=\n\nStep|\n\nFinal Answer:|$)"
        
        # Extract steps and assumptions
        matches = re.finditer(step_pattern, response, re.DOTALL)
        
        for match in matches:
            step_num = int(match.group(1))
            reasoning = match.group(2).strip()
            assumptions = match.group(3).strip() if match.group(3) else ""
            
            steps.append({
                "step_num": step_num,
                "reasoning": reasoning,
                "assumptions": assumptions
            })
        
        # Extract the final answer and confidence
        final_answer_match = re.search(r"Final Answer: (.*?)(?:\nConfidence: ([\d\.]+))?$", response, re.DOTALL)
        
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            confidence = float(final_answer_match.group(2)) if final_answer_match.group(2) else 0.5
            
            steps.append({
                "step_num": len(steps) + 1,
                "reasoning": f"Final Answer: {answer}",
                "assumptions": f"Confidence level: {confidence}",
                "is_final": True,
                "confidence": confidence
            })
        
        self.logger.info(f"Extracted {len(steps)} reasoning steps")
        return steps
    
    def extract_assumptions(self, step: Dict[str, Any]) -> List[str]:
        """
        Extract assumptions from a reasoning step, enforcing the maximum number of assumptions.
        
        Args:
            step: A dictionary containing the reasoning step with 'reasoning' and 'assumptions' keys
            
        Returns:
            A list of extracted assumptions, limited to max_assumptions if specified
        """
        if not self.assumption_extraction:
            return [step.get("assumptions", "")]
        
        assumptions_text = step.get("assumptions", "")
        
        # If no explicit assumptions, try to infer implicit ones from the reasoning text
        if not assumptions_text.strip():
            assumptions = self._infer_implicit_assumptions(step["reasoning"])
        else:
            # First try to extract numbered assumptions (more reliable)
            numbered_assumptions = re.findall(r'(?i)(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|$)', assumptions_text, re.DOTALL)
            
            if numbered_assumptions:
                assumptions = [a.strip() for a in numbered_assumptions if a.strip()]
            else:
                # Fall back to splitting by common separators
                assumptions = re.split(r'[;\n]', assumptions_text)
                assumptions = [a.strip() for a in assumptions if a.strip()]
        
        # Clean up any remaining markers or numbers in the assumptions
        assumptions = [re.sub(r'^\s*[\d\.\-]\s*', '', a).strip() for a in assumptions]
        assumptions = [a for a in assumptions if a]
        
        # Apply max_assumptions limit if set
        if self.max_assumptions is not None and self.max_assumptions > 0:
            if len(assumptions) > self.max_assumptions:
                self.logger.info(
                    f"Truncating {len(assumptions)} assumptions to {self.max_assumptions} "
                    f"for step {step.get('step_num', 'unknown')}. Original assumptions: {assumptions}"
                )
                assumptions = assumptions[:self.max_assumptions]
                self.logger.info(f"Truncated assumptions: {assumptions}")
            elif len(assumptions) < self.max_assumptions:
                self.logger.info(
                    f"Only {len(assumptions)} assumptions found for step {step.get('step_num', 'unknown')} "
                    f"(expected up to {self.max_assumptions}). Assumptions: {assumptions}"
                )
            else:
                self.logger.info(f"Step {step.get('step_num', 'unknown')} has exactly {len(assumptions)} assumptions")
        else:
            self.logger.info(f"No assumption limit set. Step {step.get('step_num', 'unknown')} has {len(assumptions)} assumptions")
        
        return assumptions
    
    def _infer_implicit_assumptions(self, reasoning: str) -> List[str]:
        """
        Infer implicit assumptions from reasoning text when not explicitly provided.
        
        Args:
            reasoning: The reasoning text to analyze for implicit assumptions
            
        Returns:
            A list of inferred assumptions, respecting max_assumptions if set
        """
        # Try to extract key phrases that might indicate assumptions
        assumption_phrases = []
        
        # Look for common assumption indicators in the text
        assumption_patterns = [
            r'(?i)(?:assum(?:e|ing|ed)|presum(?:e|ing|ed)|suppos(?:e|ing|ed))\s+(?:that\s+)?([^\.!?]+)[\.!?]',
            r'(?i)(?:if|when|assuming|given that|provided that|in case)\s+([^\.!?]+)[\.!?]',
            r'(?i)(?:this (?:suggests|implies|indicates|means)|which (?:suggests|implies|indicates|means))\s+that\s+([^\.!?]+)[\.!?]'
        ]
        
        for pattern in assumption_patterns:
            matches = re.findall(pattern, reasoning)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Take the first group if there are groups
                if match and len(match) > 10:  # Only include non-trivial matches
                    # Clean up the text
                    assumption = match.strip()
                    # Remove any leading 'that' or 'which'
                    assumption = re.sub(r'^(?:that|which|,)\s+', '', assumption, flags=re.IGNORECASE)
                    # Capitalize first letter
                    if assumption:
                        assumption = assumption[0].upper() + assumption[1:]
                        assumption_phrases.append(f"Implicit assumption: {assumption}")
        
        # # If no patterns matched, use some fallback assumptions
        # if not assumption_phrases:
        #     assumption_phrases = [
        #         "Implicit assumption: The reasoning is based on complete and accurate information.",
        #         "Implicit assumption: The context provided is sufficient for drawing conclusions.",
        #         "Implicit assumption: Standard conditions and constraints apply unless stated otherwise."
        #     ]
        
        # Apply max_assumptions limit if set
        if self.max_assumptions is not None and self.max_assumptions > 0:
            assumption_phrases = assumption_phrases[:self.max_assumptions]
        
        return assumption_phrases
    
    def annotate_with_assumptions(self, step: Dict[str, Any], assumptions: List[str]) -> Dict[str, Any]:
        """
        Annotate a reasoning step with extracted assumptions.
        
        Args:
            step: The reasoning step dictionary
            assumptions: List of extracted assumptions
            
        Returns:
            The updated reasoning step with annotated assumptions
        """
        step["extracted_assumptions"] = assumptions
        
        # Create a formatted string representation for output
        formatted_assumptions = "; ".join(assumptions)
        step["annotated_reasoning"] = f"{step['reasoning']}\n[Assumptions: {formatted_assumptions}]"
        
        return step
    
    def generate(self, query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a sequence of reasoning steps for the given query.
        
        Args:
            query: The query to reason about
            context: Optional additional context
            
        Returns:
            A list of reasoning steps with embedded assumptions
        """
        self.logger.info(f"Generating reasoning steps for query: {query}")
        
        # Construct the prompt
        prompt = self.construct_prompt(query, context)
        
        # Query the LLM
        response = self.query_llm(prompt)
        
        # Extract reasoning steps
        steps = self.extract_reasoning_steps(response)
        
        # Extract and annotate assumptions for each step
        for i, step in enumerate(steps):
            self.logger.info(f"Processing step {i+1} for assumptions extraction")
            assumptions = self.extract_assumptions(step)
            steps[i] = self.annotate_with_assumptions(step, assumptions)
            self.logger.info(f"Step {i+1} final assumptions ({len(assumptions)}): {assumptions}")
        
        self.logger.info(f"Generated {len(steps)} reasoning steps with assumptions")
        # Log the total number of assumptions across all steps
        total_assumptions = sum(len(step.get('extracted_assumptions', [])) for step in steps)
        self.logger.info(f"Total assumptions across all steps: {total_assumptions}")
        self.logger.info(f"Max assumptions per step: {self.max_assumptions if self.max_assumptions is not None else 'No limit'}")
        return steps
    
    def generate_alternative(self, steps: List[Dict[str, Any]], rejected_steps: List[Dict[str, Any]]) -> str:
        """
        Generate an alternative answer that addresses rejected reasoning steps.
        
        Args:
            steps: All reasoning steps
            rejected_steps: Steps that were rejected
            
        Returns:
            An alternative answer addressing the rejections
        """
        # Construct a prompt for generating an alternative answer
        rejected_indices = [i for i, step in enumerate(steps) if step in rejected_steps]
        
        prompt = f"""
Based on the following reasoning steps, some steps have been identified as problematic:

{chr(10).join([f"Step {step['step_num']}: {step['reasoning']}" for step in steps])}

The problematic steps are: {', '.join([f"Step {step['step_num']}" for step in rejected_steps])}

Please provide an alternative answer that addresses these issues.
"""
        
        # Query the LLM for an alternative
        response = self.query_llm(prompt)
        
        # Extract just the alternative answer (not the full reasoning)
        alternative_match = re.search(r"(Alternative Answer|Final Answer): (.*?)$", response, re.DOTALL)
        
        if alternative_match:
            return alternative_match.group(2).strip()
        else:
            return response.strip()
    
    def generate_caveated(self, steps: List[Dict[str, Any]], abstained_steps: List[Dict[str, Any]]) -> str:
        """
        Generate an answer with caveats based on uncertain steps.
        
        Args:
            steps: All reasoning steps
            abstained_steps: Steps with high uncertainty
            
        Returns:
            An answer with appropriate caveats
        """
        uncertain_indices = [i for i, step in enumerate(steps) if step in abstained_steps]
        
        prompt = f"""
Based on the following reasoning steps, some steps have high uncertainty:

{chr(10).join([f"Step {step['step_num']}: {step['reasoning']}" for step in steps])}

The uncertain steps are: {', '.join([f"Step {step['step_num']}" for step in abstained_steps])}

Please provide an answer that acknowledges this uncertainty with appropriate caveats.
"""
        
        # Query the LLM for a caveated answer
        response = self.query_llm(prompt)
        
        # Extract the caveated answer
        caveated_match = re.search(r"(Caveated Answer|Final Answer): (.*?)$", response, re.DOTALL)
        
        if caveated_match:
            return caveated_match.group(2).strip()
        else:
            return response.strip()
    
    def _simulated_response(self, prompt: str) -> str:
        """
        Generate a simulated response when the LLM API is unavailable.
        
        Args:
            prompt: The input prompt that would have been sent to the API
            
        Returns:
            A simulated response with a clear indication it's a fallback
        """
        self.logger.warning("Using simulated response as fallback (API unavailable)")
        
        # Extract the main query from the prompt (simplified extraction)
        query = prompt.split("Query:", 1)[-1].split("\n", 1)[0].strip()
        
        # Generate a simple fallback response
        return f"""
        [SIMULATED RESPONSE - API UNAVAILABLE]
        
        I'm currently unable to access the AI service. Here's a simulated response to your query:
        
        Query: {query}
        
        Step 1: Identified the main topic of the query
        - Assumption: The query is about {query.split()[0]} if it were a real request
        
        Step 2: Analyzed potential approaches
        - Assumption: Multiple perspectives could be considered
        - Assumption: Data might be available to support different views
        
        Step 3: Generated a basic response
        - Assumption: The user is looking for a thoughtful analysis
        - Assumption: A balanced view would be most helpful
        
        Final Answer: This is a simulated response because the AI service is currently unavailable. 
        Please check your API key and internet connection, then try again.
        """.strip()

    def generate_confident(self, accepted_steps: List[Dict[str, Any]]) -> str:
        """
        Generate a confident answer based on accepted steps.
        
        Args:
            accepted_steps: Steps that were accepted
            
        Returns:
            A confident answer
        """
        if not accepted_steps:
            return "Insufficient reliable information to provide an answer."
        
        # Extract the final step if it exists
        final_steps = [step for step in accepted_steps if step.get("is_final", False)]
        
        if final_steps:
            # Use the existing final step
            final_step = final_steps[0]
            return final_step["reasoning"].replace("Final Answer:", "").strip()
        else:
            # Generate a new conclusion from the accepted steps
            prompt = f"""
Based on the following reliable reasoning steps:

{chr(10).join([f"Step {step['step_num']}: {step['reasoning']}" for step in accepted_steps])}

Please provide a concise and confident conclusion.
"""
            
            response = self.query_llm(prompt)
            
            # Extract just the conclusion
            conclusion_match = re.search(r"(Conclusion|Final Answer): (.*?)$", response, re.DOTALL)
            
            if conclusion_match:
                return conclusion_match.group(2).strip()
            else:
                return response.strip()
