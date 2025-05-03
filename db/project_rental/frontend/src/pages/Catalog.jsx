import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from '../axios';
import { convertToSlug } from '../utils';

function Catalog() {
    const [categories, setCategories] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchCategories = async () => {
            try {
                const res = await axios.get('/catalog');
                setCategories(res.data);
            } catch (error) {
                console.error('Ошибка загрузки категорий:', error);
                setError('Не удалось загрузить категории. Попробуйте позже.');
            } finally {
                setLoading(false);
            }
        };
        fetchCategories();
    }, []);

    const handleCategoryClick = (categoryName) => {
        navigate(`/catalog/${convertToSlug(categoryName)}`, { state: { categoryName: categoryName } });
    };

    return (
        <div style={{ padding: '0 2rem 2rem 2rem' }}>
            <p className="subtext" style={{ margin: '0.5rem 0 0.5rem 0' }}>Каталог</p>
            <h1 className="page-title">Каталог</h1>

            {loading ? (
                <p></p>
            ) : error ? (
                <p style={{ color: 'red' }}>{error}</p>
            ) : (
                <div className="category-grid">
                    {categories.map((category) => (
                        <div key={category.name} className="category-item">
                            <button onClick={() => handleCategoryClick(category.name)} className="category-box-link">
                                <div className="category-box"></div>
                                <p>{category.name}</p>
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default Catalog;
