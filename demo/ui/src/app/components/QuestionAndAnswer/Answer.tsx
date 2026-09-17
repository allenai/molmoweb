import { Typography } from '@mui/material';

const numberFormatter = Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 2 });

export interface AnswerType {
    answer: string;
    score: number;
}

interface AnswerProps {
    answer?: AnswerType;
}

export default function Answer({ answer }: AnswerProps) {
    if (answer == null) {
        return null;
    }

    return (
        <Typography>
            The answer is: <strong>{answer.answer}</strong> ({numberFormatter.format(answer.score)}{' '}
            confidence)
        </Typography>
    );
}
