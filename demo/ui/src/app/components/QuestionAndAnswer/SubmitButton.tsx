'use client';
import { PropsWithChildren } from 'react';
import { useFormState } from 'react-hook-form-mui';
import { LoadingButton } from '@mui/lab';

export function SubmitButton({ children }: PropsWithChildren) {
    const { isSubmitting } = useFormState();

    return (
        <LoadingButton
            type="submit"
            variant="contained"
            loading={isSubmitting}
            sx={{ width: 'fit-content', paddingInline: 5 }}>
            {children}
        </LoadingButton>
    );
}
