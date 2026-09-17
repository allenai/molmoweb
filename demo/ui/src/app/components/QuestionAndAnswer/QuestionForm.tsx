'use client';

import { MaxWidthText } from '@allenai/varnish2/components';
import { Stack } from '@mui/material';
import {
    FormContainer,
    MultiSelectElement,
    SubmitHandler,
    TextFieldElement,
    useForm,
} from 'react-hook-form-mui';

import { SubmitButton } from '@/app/components/QuestionAndAnswer/SubmitButton';
import { AnswerType } from './Answer';
import { solve } from './solveAction';

const defaultValues = {
    question: '',
    choices: [] as Array<string>,
};

interface FormProps {
    onSuccess: (answer: AnswerType) => void;
}

// Since we declare 'use client' at the top of this file this is a client component.
// It'll still be server-side rendered but it'll retain all the normal React component functionality you'd expect from any other app
export function QuestionForm({ onSuccess }: FormProps) {
    const formContext = useForm({
        defaultValues,
    });

    const handleSuccess: SubmitHandler<typeof defaultValues> = async (data) => {
        console.log('This log will show up in the client logs!');

        const response = await solve(data);
        onSuccess(response);
    };

    return (
        <>
            <MaxWidthText as="p">
                Enter a question and answers below to see what answer our application selects.
            </MaxWidthText>

            <Stack maxWidth="sm" width={1}>
                <FormContainer formContext={formContext} onSuccess={handleSuccess}>
                    <Stack spacing={2}>
                        <TextFieldElement
                            name="question"
                            fullWidth
                            label="Question"
                            multiline
                            placeholder="Enter a question"
                            required
                            rows={4}
                            validation={{
                                required: 'Question is required',
                                minLength: {
                                    value: 5,
                                    message: 'Question must be at least 5 characters long',
                                },
                            }}
                        />

                        <MultiSelectElement
                            fullWidth
                            label="Answers"
                            name="choices"
                            required
                            variant="outlined"
                            options={['Grapefruit', 'Lemon', 'Lime', 'Orange']}
                        />

                        <SubmitButton>Submit</SubmitButton>
                    </Stack>
                </FormContainer>
            </Stack>
        </>
    );
}
