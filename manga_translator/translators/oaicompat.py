import re
try:
    import openai
    import openai.error
except ImportError:
    openai = None
import asyncio
import time
import os
from typing import List, Dict

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE
from .chatgpt import GPT35TurboTranslator, GPT3Translator
CONFIG = None


class OpenAICompatibleTranslator(GPT3Translator):
    _CONFIG_KEY = 'oaicompat'
    _MAX_REQUESTS_PER_MINUTE = 200
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 32000
    _TIMEOUT = 120
    _RETURN_PROMPT = False
    _INCLUDE_TEMPLATE = False
    _MESSAGE_HISTORY = []
    _MAINTAIN_HISTORY = False
    _NUM_CHAR_TO_TOKENS = 3
    _TL_CONTEXT = None

    # Token: 57+
    _CHAT_SYSTEM_TEMPLATE = (
        'You are a professional translation engine, please translate the story into a colloquial, '
        'elegant and fluent content, without referencing machine translations. '
        'You must only translate the story, never interpret it. '
        'If there is any issue in the text, output it as is.\n'
        'Translate to {to_lang}.'
    )
    _CHAT_SAMPLE = {}

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    
    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def enable_history(self) -> float:
        return self._config_get('enable_history', default=0.5)

    @property
    def chat_tl_header(self) -> str:
        return self._config_get('chat_tl_header', None)

    @property
    def tl_context(self) -> float:
        return self._config_get('tl_context', default="")

    def _read_tl_context(self, path: str) -> str:
        # check if the file exists
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return ''

    def __init__(self, check_openai_key = True):
        super().__init__(check_openai_key)
        self.token_count_current = 0
        self._MAINTAIN_HISTORY = self._config_get('enable_history', default=False)
        # self._TL_CONTEXT = self._read_tl_context("")
        print('OpenAICompatibleTranslator: _MAINTAIN_HISTORY =', self._MAINTAIN_HISTORY)
        print('OpenAICompatibleTranslator: top_p', self._config_get('top_p', 100))
        print('OpenAICompatibleTranslator: temperature', self._config_get('temperature', 0.5))


    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        system_prompt = self.chat_system_template.format(to_lang=to_lang, tl_context=self.tl_context)
        chat_sample = [
            'User:',
            self.chat_sample[to_lang][0],
            'Assistant:',
            self.chat_sample[to_lang][1],
        ] if to_lang in self.chat_sample else []
        return '\n'.join(['System:', system_prompt, *chat_sample, 'User:', prompt])

    def _format_numbered_list(self, text, format_string="<|#|>"):
        text_prefix = ''
        if self.chat_tl_header:
            text_prefix = text.split(self.chat_tl_header)[0] + self.chat_tl_header
            text = text.split(self.chat_tl_header)[-1].strip()
        
        # Replace '#' with the captured number in the format string
        formatted_string = format_string.replace("#", r"\1")
        
        pattern = rf"((?:^[^a-zA-Z\n]*\d+[^a-zA-Z\n]*.*$\n?)+)"
        def replace(match):
            numbered_list = match.group(1)
            formatted_list = re.sub(r"^[^a-zA-Z\n]*?(\d+)[^a-zA-Z\n]*(.*)$", rf"{formatted_string} \2", numbered_list, flags=re.MULTILINE)
            return formatted_list
        
        # Perform the replacement
        formatted_text = re.sub(pattern, replace, text, flags=re.MULTILINE)

        return text_prefix + formatted_text


    
    def _estimate_token_count(self, input_text: str) -> int:
        text_length = len(input_text)
        num_alphabet = sum(char.isalpha() for char in input_text)
        non_alphabet = text_length - num_alphabet
        token_estimate = num_alphabet/self._NUM_CHAR_TO_TOKENS + non_alphabet
        return token_estimate

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        system_prompt = self.chat_system_template.format(to_lang=to_lang,tl_context=self.tl_context)
        # system_prompt = system_prompt.replace('<tl_context>', 'TESTING')
        print('system_prompt:', system_prompt)
        messages = [{'role': 'system', 'content': system_prompt}]
        # Add messages from history if enabled
        if self.enable_history:
            self.logger.debug('Adding message history to prompt...')
            # Estimate the total token length of the prompt, assuming we are only adding a new prompt to previous request
            total_length_tokens = self.token_count_current + self._estimate_token_count(prompt)
            # If total length exceeds the max tokens, remove the oldest messages until it fits
            while len(self._MESSAGE_HISTORY) > 0 and total_length_tokens > self._MAX_TOKENS:
                removed_message = self._MESSAGE_HISTORY.pop(0)
                self.logger.debug(f'Removing oldest message from history: {removed_message["content"]}')
                total_length_tokens -= self._estimate_token_count(removed_message["content"])
                # self.token_count_current -= float(removed_message['tokens'])
                self.logger.debug(f'Previous Request Token Count: {self.token_count_last} tokens, Current Token Count: {total_length_tokens} tokens')
            for message in self._MESSAGE_HISTORY:
                messages.append(message)
            self.logger.debug(f'Added {len(self._MESSAGE_HISTORY)} messages to prompt. Total length: {total_length_tokens} tokens (est. prompt {self._NUM_CHAR_TO_TOKENS} chars per token).')
        # Add the prompt as a user message
        messages.append({'role': 'user', 'content': prompt})

        if to_lang in self._CHAT_SAMPLE:
            messages.insert(1, {'role': 'user', 'content': self._CHAT_SAMPLE[to_lang][0]})
            messages.insert(2, {'role': 'assistant', 'content': self._CHAT_SAMPLE[to_lang][1]})

        response = await openai.ChatCompletion.acreate(
            model='local-llm',
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=0.05,
        )

        self.token_count += response.usage['total_tokens']
        self.token_count_last = response.usage['total_tokens']
        if self.token_count_last > 0:
            self.token_count_current = response.usage['total_tokens']
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- LLM Prompt --\n' + self._format_prompt_log(to_lang, prompt))

            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            while True:
                request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                started = time.time()
                while not request_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                        # Server takes too long to respond
                        if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                            raise Exception('openai servers did not respond quickly enough.')
                        timeout_attempt += 1
                        self.logger.warn(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                        request_task.cancel()
                        request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                        started = time.time()
                try:
                    response = await request_task
                    break
                except openai.error.RateLimitError: # Server returned ratelimit response
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                        raise
                    self.logger.warn(f'Restarting request due to ratelimiting by openai servers. Attempt: {ratelimit_attempt}')
                    await asyncio.sleep(2)
                except openai.error.APIError: # Server returned 500 error (probably server load)
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error('OpenAI encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)

            self.logger.debug('-- LLM Response --\n' + response)

            # Ensure the response list is formatted correctly
            response = self._format_numbered_list(response)
            tl_response = response
            # If set, split the response by the chat_tl_header and take the last portion
            if self.chat_tl_header:
                tl_response = response.split(self.chat_tl_header)[-1].strip()
            # Extract the list portion of the response
            
            list_pattern = r'<\|\d+\|>.*?(?=<\|\d+\|>|$)'
            response_list = re.findall(list_pattern, tl_response, re.DOTALL)
            new_translations = [re.sub(r'<\|\d+\|>', '', item).strip() for item in response_list]
            # Fill in blank new_translations with the original response
            new_translations = ['ERROR' if not item else item for item in new_translations]
            self.logger.debug(f'-- LLM Translations --\n{new_translations}')

            # When there is only one query chatgpt likes to exclude the <|1|>
            if len(new_translations) > 0 and not new_translations[0].strip():
                new_translations = new_translations[1:]

            if len(new_translations) <= 1 and query_size > 1:
                # Try splitting by newlines instead
                new_translations = re.split(r'\n', response)

            if len(new_translations) != query_size:
                # super method will repeat translation as per self._INVALID_REPEAT_COUNT
                translations = []
                break
            
            # Update message history
            self._MESSAGE_HISTORY.append({'role': 'user', 'content': prompt})
            self._MESSAGE_HISTORY.append({'role': 'assistant', 'content': response})

            translations.extend([t.strip() for t in new_translations])

        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

