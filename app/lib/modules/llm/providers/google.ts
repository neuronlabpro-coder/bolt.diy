--- START OF FILE google.ts ---

import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

export default class GoogleProvider extends BaseProvider {
  name = 'Google';
  getApiKeyLink = 'https://aistudio.google.com/app/apikey';

  config = {
    apiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
  };

  staticModels: ModelInfo[] = [
    // Gemini 3 Series
    {
      name: 'gemini-3-pro-preview',
      label: 'Gemini 3 Pro Preview',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 65536, // Gemini 3 soporta outputs largos
    },

    // Gemini 2.5 Series
    {
      name: 'gemini-2.5-pro',
      label: 'Gemini 2.5 Pro',
      provider: 'Google',
      maxTokenAllowed: 2097152,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-2.5-flash',
      label: 'Gemini 2.5 Flash',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-2.5-flash-lite',
      label: 'Gemini 2.5 Flash Lite',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },

    // Gemini 2.0 Series
    {
      name: 'gemini-2.0-flash',
      label: 'Gemini 2.0 Flash',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },

    // Latest Aliases
    {
      name: 'gemini-flash-latest',
      label: 'Gemini Flash (Latest)',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-flash-lite-latest',
      label: 'Gemini Flash Lite (Latest)',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },

    // Fallback Stable 1.5 Series
    {
      name: 'gemini-1.5-pro',
      label: 'Gemini 1.5 Pro',
      provider: 'Google',
      maxTokenAllowed: 2097152,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-1.5-flash',
      label: 'Gemini 1.5 Flash',
      provider: 'Google',
      maxTokenAllowed: 1048576,
      maxCompletionTokens: 8192,
    },
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv?: Record<string, string>,
  ): Promise<ModelInfo[]> {
    const { apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: '',
      defaultApiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
    });

    if (!apiKey) {
      throw `Missing Api Key configuration for ${this.name} provider`;
    }

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`, {
      headers: {
        ['Content-Type']: 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch models from Google API: ${response.status} ${response.statusText}`);
    }

    const res = (await response.json()) as any;

    if (!res.models || !Array.isArray(res.models)) {
      throw new Error('Invalid response format from Google API');
    }

    // Filter logic: Allow stable, flash, and specific preview/experimental models requested
    const data = res.models.filter((model: any) => {
      const name = model.name.toLowerCase();
      
      // Permitir explícitamente modelos 2.0, 2.5, 3.0, flash y pro
      const isNewGen = name.includes('gemini-2') || name.includes('gemini-3');
      const isFlash = name.includes('flash');
      const isPro = name.includes('pro');
      
      // Filtro base: token output decente Y (es estable O es una de las nuevas previews permitidas)
      const hasGoodTokenLimit = (model.outputTokenLimit || 0) >= 8000;
      
      // Permitir previews si son de las generaciones nuevas
      const isAllowedPreview = (name.includes('preview') || name.includes('exp')) && isNewGen;
      const isStable = !name.includes('exp') && !name.includes('preview');

      return hasGoodTokenLimit && (isStable || isAllowedPreview || isFlash || isPro);
    });

    return data.map((m: any) => {
      const modelName = m.name.replace('models/', '');

      // Context Window Logic
      let contextWindow = 32000; // default safe fallback

      if (m.inputTokenLimit) {
        contextWindow = m.inputTokenLimit;
      } else {
        // Fallback heuristics based on name if API doesn't return limit
        if (modelName.includes('gemini-3')) {
          contextWindow = 1048576; // 1M
        } else if (modelName.includes('gemini-2.5-pro') || modelName.includes('gemini-1.5-pro')) {
          contextWindow = 2097152; // 2M
        } else if (modelName.includes('gemini-2') || modelName.includes('flash')) {
          contextWindow = 1048576; // 1M standard for Flash/2.0
        }
      }

      // Cap at reasonable limits
      const maxAllowed = 2097152; // 2M cap
      const finalContext = Math.min(contextWindow, maxAllowed);

      // Completion Token Logic
      let completionTokens = 8192;

      if (m.outputTokenLimit) {
        completionTokens = m.outputTokenLimit;
      } else if (modelName.includes('gemini-3')) {
        completionTokens = 65536; // Gemini 3 supports higher output
      }

      // Ensure we don't return an unreasonably small completion limit unless it's real
      if (completionTokens < 4096) completionTokens = 4096;

      return {
        name: modelName,
        label: `${m.displayName || modelName} (${Math.floor(finalContext / 1000)}k context)`,
        provider: this.name,
        maxTokenAllowed: finalContext,
        maxCompletionTokens: completionTokens,
      };
    });
  }

  getModelInstance(options: {
    model: string;
    serverEnv: any;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: '',
      defaultApiTokenKey: 'GOOGLE_GENERATIVE_AI_API_KEY',
    });

    if (!apiKey) {
      throw new Error(`Missing API key for ${this.name} provider`);
    }

    const google = createGoogleGenerativeAI({
      apiKey,
    });

    return google(model);
  }
}
