#!/usr/bin/env python3
"""
Baseline LLM prompting script - direct prompting without ReAct framework.
Supports Anthropic, OpenAI, and Gemini APIs.
Used for comparison against the ReAct agent.
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prompt_anthropic(prompt: str, config: dict) -> str:
    """Send prompt to Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    settings = config["anthropic"]

    message = client.messages.create(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def prompt_openai(prompt: str, config: dict) -> str:
    """Send prompt to OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    settings = config["openai"]

    response = client.chat.completions.create(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def prompt_gemini(prompt: str, config: dict) -> str:
    """Send prompt to Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    settings = config["gemini"]

    model = genai.GenerativeModel(settings["model"])
    generation_config = genai.GenerationConfig(
        max_output_tokens=settings["max_tokens"],
        temperature=settings["temperature"]
    )

    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text


def log_response(
    provider: str,
    model: str,
    prompt: str,
    response: str,
    config: dict
) -> str:
    """Log response to file organized by provider with timestamp."""
    log_config = config["logging"]
    output_dir = Path(log_config["output_dir"]) / provider
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "provider": provider,
        "model": model,
        "mode": "baseline",
        "task": prompt,
        "response": response
    }

    if log_config["format"] == "json":
        filename = f"base_{timestamp_str}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)
    else:
        filename = f"base_{timestamp_str}.txt"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            f.write(f"Timestamp: {timestamp.isoformat()}\n")
            f.write(f"Provider: {provider}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Mode: baseline\n")
            f.write(f"Task:\n{prompt}\n")
            f.write(f"\nResponse:\n{response}\n")

    return str(filepath)


def prompt_llm(prompt: str, config_path: str = "config.yaml") -> dict:
    """
    Send prompt to configured LLM provider and log response.

    Returns dict with provider, model, response, and log_file.
    """
    config = load_config(config_path)
    provider = config["provider"]

    providers = {
        "anthropic": prompt_anthropic,
        "openai": prompt_openai,
        "gemini": prompt_gemini
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    model = config[provider]["model"]
    response = providers[provider](prompt, config)
    log_file = log_response(provider, model, prompt, response, config)

    return {
        "provider": provider,
        "model": model,
        "response": response,
        "log_file": log_file
    }


def main():
    """Interactive prompt loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline LLM prompting (no ReAct framework)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-p", "--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("-f", "--file", help="Read prompt from file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Using provider: {config['provider']} ({config[config['provider']]['model']})")
    print("Mode: Baseline")

    # Read prompt from file if specified
    if args.file:
        with open(args.file, "r") as f:
            prompt = f.read().strip()
        result = prompt_llm(prompt, args.config)
        print(f"\nResponse:\n{result['response']}")
        print(f"\nLogged to: {result['log_file']}")
    elif args.prompt:
        result = prompt_llm(args.prompt, args.config)
        print(f"\nResponse:\n{result['response']}")
        print(f"\nLogged to: {result['log_file']}")
    else:
        print("Enter prompts (Ctrl+C to exit):\n")
        while True:
            try:
                prompt = input("> ")
                if not prompt.strip():
                    continue
                result = prompt_llm(prompt, args.config)
                print(f"\n{result['response']}\n")
                print(f"[Logged to: {result['log_file']}]\n")
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
