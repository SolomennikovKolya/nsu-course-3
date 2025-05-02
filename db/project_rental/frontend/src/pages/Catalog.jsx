import React, { useEffect, useState } from 'react';
import axios from '../axios';

function Catalog() {
    const [categories, setCategories] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchCategories = async () => {
            try {
                const res = await axios.get('/catalog');
                setCategories(res.data);
            } catch (error) {
                console.error('Ошибка загрузки категорий:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchCategories();
    }, []);

    return (
        <div style={{ padding: '2rem' }}>
            <h1 className="page-title">Каталог</h1>
            <p className="subtext">каталог</p>

            {/* Сетка категорий */}
            {loading ? (
                <p>Загрузка категорий...</p>
            ) : (
                <div className="category-grid">
                    {categories.map((category, index) => (
                        <div key={index} className="category-item">
                            <div className="category-box"></div>
                            <p>{category.name}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default Catalog;
