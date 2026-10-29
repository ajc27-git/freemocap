import {createAsyncThunk} from '@reduxjs/toolkit';
import {serverUrls} from '@/services';
import {getDetailedErrorMessage} from '@/store/slices/thunk-helpers';
import {splitParentAndName} from '@/store/slices/active-recording';

/**
 * The public datasets the backend can download from GitHub and extract into the
 * recordings folder. See `freemocap/utilities/download_sample_data.py`.
 */
export type DatasetKey = 'test' | 'sample';

export interface DownloadedDataset {
    datasetKey: DatasetKey;
    recordingName: string;
    recordingPath: string;
    /** Parent folder of the recording (the recordings folder), used as the active-recording base. */
    baseDirectory: string | null;
    /** False when the dataset was already on disk and no download was needed. */
    downloaded: boolean;
}

interface DatasetDownloadResponse {
    success: boolean;
    dataset_key: string;
    recording_name: string;
    recording_path: string;
    recordings_folder_path: string;
    downloaded: boolean;
}

export const downloadDataset = createAsyncThunk<
    DownloadedDataset,
    { datasetKey: DatasetKey; force?: boolean },
    { rejectValue: string }
>(
    'datasets/download',
    async ({datasetKey, force}: { datasetKey: DatasetKey; force?: boolean }, {rejectWithValue}) => {
        try {
            const response = await fetch(serverUrls.endpoints.downloadDataset(datasetKey, force), {
                method: 'POST',
            });

            if (!response.ok) {
                return rejectWithValue(await getDetailedErrorMessage(response));
            }

            const result = (await response.json()) as DatasetDownloadResponse;
            const parsed = splitParentAndName(result.recording_path ?? '');
            return {
                datasetKey,
                recordingName: result.recording_name,
                recordingPath: result.recording_path,
                baseDirectory: result.recordings_folder_path || parsed?.baseDirectory || null,
                downloaded: result.downloaded,
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error(`❌ Failed to download dataset '${datasetKey}':`, errorMessage);
            return rejectWithValue(errorMessage);
        }
    }
);
