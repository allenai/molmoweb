'use client';

import { useState } from 'react';

import Answer, { AnswerType } from './Answer';
import { QuestionForm } from './QuestionForm';

export default function QuestionAndAnswer() {
    const [answer, setAnswer] = useState<AnswerType>();

    const handleSuccess = (answer: AnswerType) => {
        setAnswer(answer);
    };

    return (
        <>
            <QuestionForm onSuccess={handleSuccess} />
            <Answer answer={answer} />
        </>
    );
}
