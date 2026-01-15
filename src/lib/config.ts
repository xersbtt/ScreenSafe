/**
 * ScreenSafe - Configuration File Management
 * 
 * Handles loading and saving app settings to a proper config file
 * in the user's app data directory (instead of localStorage).
 * 
 * Location:
 * - Windows: C:\Users\<user>\AppData\Roaming\com.screensafe.app\config.json
 * - macOS: ~/Library/Application Support/com.screensafe.app/config.json
 * - Linux: ~/.config/com.screensafe.app/config.json
 */

import { appDataDir, join } from '@tauri-apps/api/path';
import { readTextFile, writeTextFile, mkdir, exists, BaseDirectory } from '@tauri-apps/plugin-fs';

export interface AppSettings {
    theme: 'dark' | 'light';
    blurStrength: number;
    autoSaveProjects: boolean;
    ocrLanguage: 'eng' | 'auto';
    exportQuality: 'low' | 'medium' | 'high';
    exportCodec: 'h264' | 'h265' | 'vp9';
    // Scan/export processing settings
    scanInterval: number;
    motionThreshold: number;
    ocrScale: number;
    // Detection settings
    enableRegexPatterns: boolean;
}

const CONFIG_FILENAME = 'config.json';

const DEFAULT_SETTINGS: AppSettings = {
    theme: 'dark',
    blurStrength: 25,
    autoSaveProjects: false,
    ocrLanguage: 'eng',
    exportQuality: 'high',
    exportCodec: 'h264',
    scanInterval: 30,
    motionThreshold: 30,
    ocrScale: 0.75,
    enableRegexPatterns: false
};

/**
 * Get the config file path using Tauri's path API
 */
async function getConfigPath(): Promise<string> {
    const dataDir = await appDataDir();
    return await join(dataDir, CONFIG_FILENAME);
}

/**
 * Ensure the config directory exists
 */
async function ensureConfigDir(): Promise<void> {
    try {
        // Use BaseDirectory.AppData to let Tauri handle the path
        const dirExists = await exists('', { baseDir: BaseDirectory.AppData });
        if (!dirExists) {
            await mkdir('', { baseDir: BaseDirectory.AppData, recursive: true });
            console.log('[Config] Created app data directory');
        }
    } catch (err) {
        console.warn('[Config] Could not verify/create config directory:', err);
        // Try to create using full path as fallback
        try {
            const dataDir = await appDataDir();
            await mkdir(dataDir, { recursive: true });
        } catch (fallbackErr) {
            console.warn('[Config] Fallback mkdir also failed:', fallbackErr);
        }
    }
}

/**
 * Load settings from config file
 * Falls back to defaults if file doesn't exist or is invalid
 */
export async function loadSettings(): Promise<AppSettings> {
    try {
        // Try using BaseDirectory first (cleaner approach)
        try {
            const fileExists = await exists(CONFIG_FILENAME, { baseDir: BaseDirectory.AppData });
            if (fileExists) {
                const content = await readTextFile(CONFIG_FILENAME, { baseDir: BaseDirectory.AppData });
                const parsed = JSON.parse(content);
                const settings = { ...DEFAULT_SETTINGS, ...parsed };
                console.log('[Config] Loaded settings from AppData');
                return settings;
            }
        } catch (baseErr) {
            console.log('[Config] BaseDirectory approach failed, trying full path:', baseErr);
        }

        // Fallback to full path approach
        const configPath = await getConfigPath();
        console.log('[Config] Trying full path:', configPath);

        const fileExists = await exists(configPath);
        if (!fileExists) {
            console.log('[Config] No config file found, using defaults');
            return { ...DEFAULT_SETTINGS };
        }

        const content = await readTextFile(configPath);
        const parsed = JSON.parse(content);

        // Merge with defaults to handle missing keys from older versions
        const settings = { ...DEFAULT_SETTINGS, ...parsed };
        console.log('[Config] Loaded settings from:', configPath);
        return settings;
    } catch (err) {
        console.warn('[Config] Failed to load settings, using defaults:', err);
        return { ...DEFAULT_SETTINGS };
    }
}

/**
 * Save settings to config file
 */
export async function saveSettings(settings: AppSettings): Promise<void> {
    const content = JSON.stringify(settings, null, 2);

    try {
        // Try using BaseDirectory first (cleaner approach)
        await ensureConfigDir();
        await writeTextFile(CONFIG_FILENAME, content, { baseDir: BaseDirectory.AppData });
        console.log('[Config] Settings saved to AppData');
        return;
    } catch (baseErr) {
        console.log('[Config] BaseDirectory save failed, trying full path:', baseErr);
    }

    // Fallback to full path approach
    try {
        const configPath = await getConfigPath();
        await writeTextFile(configPath, content);
        console.log('[Config] Settings saved to:', configPath);
    } catch (err) {
        console.error('[Config] Failed to save settings:', err);
        // Fallback to localStorage as backup
        try {
            localStorage.setItem('screensafe-settings-backup', JSON.stringify(settings));
            console.log('[Config] Saved to localStorage as backup');
        } catch {
            // Ignore localStorage errors
        }
    }
}

/**
 * Get default settings
 */
export function getDefaultSettings(): AppSettings {
    return { ...DEFAULT_SETTINGS };
}

