"""Fixed replies (no LLM): REPLIES[intent][language]."""

REPLIES = {
    "greeting": {
        "az": (
            "Salam! Mən UNEC-in virtual köməkçisiyəm. Qəbul, təhsil, imtahanlar, "
            "ixtisaslar və tələbə həyatı ilə bağlı suallarınıza cavab verə bilərəm. "
            "Sizə necə kömək edim?"
        ),
        "ru": (
            "Здравствуйте! Я виртуальный помощник UNEC. Могу ответить на вопросы о "
            "поступлении, обучении, экзаменах, специальностях и студенческой жизни. "
            "Чем могу помочь?"
        ),
        "en": (
            "Hello! I'm UNEC's virtual assistant. I can answer questions about admission, "
            "studies, exams, programs and student life. How can I help you?"
        ),
    },
    "thanks": {
        "az": "Rica edirəm! Başqa sualınız olsa, buradayam.",
        "ru": "Пожалуйста! Если появятся вопросы — я здесь.",
        "en": "You're welcome! I'm here if you have more questions.",
    },
    "off_topic": {
        "az": (
            "Bağışlayın, mən yalnız UNEC və orada təhsillə bağlı suallara cavab vermək "
            "üçün hazırlanmışam. Məsələn, qəbul qaydaları, ixtisaslar və ya imtahanlar "
            "haqqında soruşa bilərsiniz."
        ),
        "ru": (
            "Извините, я создан только для вопросов о UNEC и обучении в нём. Например, "
            "можете спросить о правилах поступления, специальностях или экзаменах."
        ),
        "en": (
            "Sorry, I'm designed to answer questions only about UNEC and studying there — "
            "for example admission rules, programs or exams."
        ),
    },
    "not_found": {
        "az": (
            "Bu barədə məlumat bazamda məlumat tapa bilmədim. Dəqiq məlumat üçün "
            "unec.edu.az saytına baxmağı və ya universitetin müvafiq şöbəsinə müraciət "
            "etməyi tövsiyə edirəm."
        ),
        "ru": (
            "К сожалению, я не нашёл информации об этом в своей базе. Для точной "
            "информации рекомендую посетить сайт unec.edu.az или обратиться в "
            "соответствующий отдел университета."
        ),
        "en": (
            "I couldn't find information about this in my knowledge base. For accurate "
            "information, please check unec.edu.az or contact the relevant university office."
        ),
    },
}
