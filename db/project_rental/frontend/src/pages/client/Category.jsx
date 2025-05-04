import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams, NavLink } from 'react-router-dom';
import clsx from "clsx";
import axios from '../../axios';
import { convertToSlug } from '../../utils';
import styles from './Category.module.css';

function Category() {
    const { categoryNameSlug } = useParams();               // Название категории транслитом
    const [categoryName, setCategoryName] = useState(null); // Название категории (не транслитом)
    const [categories, setCategories] = useState([]);       // Все категории
    const [equipment, setEquipment] = useState([]);         // Оборудование в данной категории
    const [loading, setLoading] = useState(true);           // Идёт ли сейчас загрузка
    const [error, setError] = useState(null);               // Ошибка
    const navigate = useNavigate();

    const getOriginalCategoryName = (slug) => {
        const category = categories.find(c => convertToSlug(c.name) === slug);
        return category ? category.name : slug;
    };

    // При первой загрузки данной страницы, дастём оригинальное название категории из state
    const location = useLocation();
    useEffect(() => {
        if (location.state?.categoryName) {
            setCategoryName(location.state.categoryName);
        }
    }, [location.state]);

    // Загрузка списка всех категорий (выполняется 1 раз при монтировании)
    useEffect(() => {
        const fetchCategories = async () => {
            try {
                const res = await axios.get('/catalog');
                setCategories(res.data);
            } catch (error) {
                console.error('Ошибка загрузки категорий:', error);
                setError('Не удалось загрузить категории.');
            }
        };
        fetchCategories();
    }, []);

    // При выборе другой категории меняем оригинальное название
    useEffect(() => {
        if (categories.length > 0) {
            setCategoryName(getOriginalCategoryName(categoryNameSlug));
        }
    }, [categories, categoryNameSlug]);

    // При выборе другой категории (а именно после изменения оригинального названия) обновляется контент
    useEffect(() => {
        if (!categoryName) return;
        const fetchEquipment = async () => {
            try {
                const res = await axios.get('/category', {
                    params: { name: categoryName }
                });
                setEquipment(res.data);
            } catch (error) {
                console.error('Ошибка загрузки оборудования:', error);
                setError('Не удалось загрузить оборудование для этой категории.');
            } finally {
                setLoading(false);
            }
        };
        fetchEquipment();
    }, [categoryName]);

    const handleEquipmentClick = (categoryName, equipmentName) => {
        navigate(`/catalog/${convertToSlug(categoryName)}/${convertToSlug(equipmentName)}`, {
            state: { categoryName: categoryName, equipmentName: equipmentName }
        });
    };

    return (
        <div style={{ padding: '0 2rem 2rem 2rem' }}>
            <p className="subtext" style={{ margin: '0.5rem 0 0.5rem 0' }}>Каталог → {categoryName}</p>
            <h1 className="page-title">{categoryName} </h1>

            {loading ? (
                <p></p>
            ) : error ? (
                <p className='error-message'>{error}</p>
            ) : (
                <div style={{ display: 'flex' }}>
                    {/* Левая панель (Навигация)*/}
                    <div style={{ width: '250px', paddingRight: '2rem' }}>
                        <ul className={styles['ul']} style={{ listStyleType: 'none', paddingLeft: 0, paddingTop: 0 }}>
                            <li className={styles['li']}>
                                <NavLink to="/catalog" className="link-style">Каталог</NavLink>
                            </li>
                            {categories.map((category) => (
                                <li className={styles['li']} key={category.name} style={{ paddingLeft: '1rem' }}>
                                    <NavLink to={`/catalog/${convertToSlug(category.name)}`} className={({ isActive }) => clsx("link-style", { "active-link": isActive })}>
                                        {category.name}
                                    </NavLink>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Правая панель (Список оборудования)*/}
                    <div style={{ flexGrow: 1 }}>
                        {loading ? (
                            <p>Загрузка оборудования...</p>
                        ) : error ? (
                            <p className='error-message'>{error}</p>
                        ) : (
                            <div className={styles["equipment-grid"]}>
                                {equipment.map((unit) => (
                                    <div key={unit.name} className={styles["equipment-item"]}>
                                        <button onClick={() => handleEquipmentClick(categoryName, unit.name)} className={styles['equipment-box-link']}>
                                            <div className={styles["equipment-box"]}></div>
                                            <p>{unit.name}<br />{unit.rental_price_per_day} ₽/день</p>
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Category;
